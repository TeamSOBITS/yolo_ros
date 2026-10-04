from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)
from yolo_ros.gui.catalog import (
    class_label,
    describe,
    IMAGE_SIZES,
    is_yoloe_model,
    sort_models,
)
from yolo_ros.gui.qt_utils import card, label
from yolo_ros.gui.widgets import (
    CheckGrid,
    compact_combo,
    page_layout,
    segment,
    segment_group,
    set_combo_items,
)


def spin(value, minimum, maximum, step):
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setDecimals(2)
    box.setValue(float(value))
    return box


class DetectPage(QWidget):
    """Model, track/detect, thresholds, input size, precision, and what to detect."""

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.system = node.system
        self.models = []
        settings = self.system.settings
        layout = page_layout(self)

        layout.addWidget(label("モデル", "Section"))
        self.model_combo = compact_combo()
        self.model_combo.activated.connect(self._model_chosen)
        self.model_note = label("", "Subtitle")
        self.track_button = segment("Track（ID を付ける）")
        self.detect_button = segment("Detect（検出だけ）")
        self.track_button.clicked.connect(lambda: self.system.set_yolo_mode("track"))
        self.detect_button.clicked.connect(lambda: self.system.set_yolo_mode("detect"))
        layout.addWidget(
            card(
                label("重みファイル（yolo_node の weights_path にあるもの）", "Subtitle"),
                self.model_combo,
                self.model_note,
                label("モード", "Subtitle"),
                segment_group(self, self.track_button, self.detect_button),
                label(
                    "Track は person:1 のように ID を付けます。Detect に切り替えても"
                    "トラッカーは残るので、Track に戻すと同じ人は同じ ID のままです。",
                    "Subtitle",
                ),
            )
        )

        layout.addWidget(label("推論", "Section"))
        self.conf_spin = spin(settings.conf, 0.05, 0.95, 0.05)
        self.conf_spin.editingFinished.connect(
            lambda: self.system.set_inference(conf=self.conf_spin.value())
        )
        self.iou_spin = spin(settings.iou, 0.1, 0.95, 0.05)
        self.iou_spin.editingFinished.connect(
            lambda: self.system.set_inference(iou=self.iou_spin.value())
        )
        self.imgsz_combo = compact_combo()
        sizes = list(IMAGE_SIZES)
        if int(settings.imgsz) not in sizes:
            sizes = sorted(sizes + [int(settings.imgsz)])
        self.sizes = sizes
        set_combo_items(
            self.imgsz_combo,
            [f"{size} px" + ("（標準）" if size == 640 else "") for size in sizes],
            sizes.index(int(settings.imgsz)),
        )
        self.imgsz_combo.activated.connect(
            lambda i: self.system.set_inference(imgsz=self.sizes[i])
        )
        self.half_check = QCheckBox("半精度（FP16）で推論する")
        self.half_check.setChecked(bool(settings.half))
        self.half_check.toggled.connect(lambda on: self.system.set_inference(half=on))
        layout.addWidget(
            card(
                label("信頼度の下限（conf）", "Subtitle"),
                self.conf_spin,
                label("重なった枠をまとめる IoU（iou）", "Subtitle"),
                self.iou_spin,
                label("入力サイズ（imgsz）", "Subtitle"),
                self.imgsz_combo,
                label(
                    "大きくすると遠くの小さな物を見つけやすくなりますが、推論時間はおよそ面積に"
                    "比例して増えます（960 で 640 の約 2 倍）。「起動」タブの推論 ms で確かめられます。",
                    "Subtitle",
                ),
                self.half_check,
                label(
                    "GPU（CUDA）で速くなり、精度はほとんど変わりません。CPU では効果がありません。"
                    "切り替えるとトラッキング ID は振り直されます。",
                    "Subtitle",
                ),
            )
        )

        layout.addWidget(label("検出する物体", "Section"))
        self.class_grid = CheckGrid(columns=2)
        self.class_grid.changed.connect(self._classes_changed)
        self.class_names = []
        buttons = QHBoxLayout()
        for text, classes in (("すべて", []), ("人だけ", ["person"])):
            button = QPushButton(text)
            button.clicked.connect(lambda _, c=classes: self._set_classes(c))
            buttons.addWidget(button)
        self.class_note = label("", "Subtitle")
        self.class_card = card(
            buttons,
            self.class_note,
            self.class_grid,
            label(
                "チェックを外した物体は枠を出さず、トラッキングもしません（filter_classes）。",
                "Subtitle",
            ),
        )
        layout.addWidget(self.class_card)

        self.prompt_edit = QLineEdit(", ".join(settings.yoloe_prompts))
        self.prompt_edit.setPlaceholderText("例: person, bottle, red cup")
        self.prompt_edit.editingFinished.connect(self._prompts_changed)
        self.prompt_card = card(
            label("YOLOE で探す物（カンマ区切り）", "Subtitle"),
            self.prompt_edit,
            label(
                "YOLOE は学習していない物でも名前（英語）で探せます。実行中にそのまま切り替わり、"
                "モデルの読み込み直しは要りません（yoloe_prompts）。",
                "Subtitle",
            ),
        )
        layout.addWidget(self.prompt_card)

        self.detail = label("", "Subtitle")
        layout.addWidget(self.detail)
        layout.addStretch()

    # ------------------------------------------------------------------ callbacks

    def _model_chosen(self, index):
        model = self.models[index]
        if not describe(model).usable:
            self.node.log("warn", f"{model} は yolo_node では使えません: {describe(model).description}")
            return
        self.system.set_yolo_model(model)

    def _set_classes(self, classes):
        self.system.set_classes(classes)
        self._update_classes()

    def _classes_changed(self, selected):
        # Everything checked means no filter (also covers classes of a future model).
        self.system.set_classes([] if len(selected) == len(self.class_names) else selected)
        self._update_classes()

    def _prompts_changed(self):
        prompts = [text.strip() for text in self.prompt_edit.text().split(",") if text.strip()]
        if not prompts:
            self.prompt_edit.setText(", ".join(self.system.settings.yoloe_prompts))
            return
        if prompts != list(self.system.settings.yoloe_prompts):
            self.system.set_yoloe_prompts(prompts)

    # ------------------------------------------------------------------ refresh

    def _update_classes(self):
        names = self.node.model_info().get("classes", [])
        chosen = list(self.system.settings.classes)
        if names:
            self.class_names = names
            self.class_grid.set_items({name: class_label(name) for name in names}, chosen or names)
        if not names:
            text = "YOLO を起動すると、そのモデルが検出できる物体の一覧が出ます。"
        elif not chosen:
            text = f"今: すべて（{len(names)} 種類）"
        else:
            text = "今: " + "、".join(class_label(name) for name in chosen)
        unknown = [name for name in chosen if names and name not in names]
        if unknown:
            text += f"\nこのモデルにない物体: {', '.join(unknown)}"
        self.class_note.setText(text)

    def update_status(self, status):
        settings = self.system.settings
        self.models = sort_models(self.system.yolo_models())
        current_model = status.yolo_model or settings.yolo_model
        if not self.model_combo.view().isVisible():
            set_combo_items(
                self.model_combo,
                [describe(model).title for model in self.models],
                self.models.index(current_model) if current_model in self.models else -1,
            )
            for index, model in enumerate(self.models):
                self.model_combo.model().item(index).setEnabled(describe(model).usable)
        if self.models:
            shown = self.models[max(0, self.model_combo.currentIndex())]
            self.model_note.setText(describe(shown).description)
        else:
            self.model_note.setText("YOLO のノードにつながると、重みファイルの一覧が出ます。")

        yoloe = is_yoloe_model(current_model)
        self.class_card.setVisible(not yoloe)
        self.prompt_card.setVisible(yoloe)
        if not yoloe:
            self._update_classes()

        ready = status.yolo_state in ("active", "inactive") and not status.busy
        self.model_combo.setEnabled(ready)
        mode = status.yolo_mode or settings.yolo_mode
        self.track_button.setChecked(mode == "track")
        self.detect_button.setChecked(mode == "detect")

        if status.busy:
            self.detail.setText(status.busy)
        elif status.yolo_state is None:
            self.detail.setText("YOLO のノードがありません。")
        elif status.yolo_state != "active":
            self.detail.setText("YOLO は OFF です。設定は「起動」したときに反映されます。")
        else:
            self.detail.setText("モデルの変更は、YOLO を一瞬止めて読み込み直します。")
