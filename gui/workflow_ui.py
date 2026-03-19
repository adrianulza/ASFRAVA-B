import logging
from tkinter import messagebox

import customtkinter as ctk

logger = logging.getLogger(__name__)


class WorkflowSetupDialog(ctk.CTkToplevel):
    """Modal workflow dialog shown before the main app is revealed."""

    def __init__(self, parent) -> None:
        super().__init__(parent)

        self.parent = parent
        self.result: dict | None = None

        self.title("Workflow Setup")
        self.geometry("980x680")
        self.minsize(900, 620)

        self.selected_method = ctk.StringVar(value="PGA")
        self.sat_period_mode = ctk.StringVar(value="calculated")
        self.saavg_period_mode = ctk.StringVar(value="calculated")
        self.sat_period_value = ctk.StringVar(value="")
        self.saavg_period_value = ctk.StringVar(value="")

        self._build_layout()
        self._refresh_cards()

        self.transient(parent)
        self.grab_set()
        self.focus_force()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(24, 12))
        header.grid_columnconfigure(0, weight=1)

        self.label_title = ctk.CTkLabel(
            header,
            text="Workflow Setup",
            font=("Inter", 28, "bold"),
            anchor="w",
        )
        self.label_title.grid(row=0, column=0, sticky="w")

        self.label_subtitle = ctk.CTkLabel(
            header,
            text="Select one IM scaling method to continue",
            font=("Inter", 14),
            anchor="w",
            text_color="#4b5563",
        )
        self.label_subtitle.grid(row=1, column=0, sticky="w", pady=(6, 0))

        self.cards_host = ctk.CTkFrame(self, fg_color="transparent")
        self.cards_host.grid(row=2, column=0, sticky="nsew", padx=28, pady=(8, 16))
        self.cards_host.grid_columnconfigure((0, 1, 2), weight=1, uniform="cards")
        self.cards_host.grid_rowconfigure(0, weight=1)

        self.card_pga = self._build_pga_card(self.cards_host)
        self.card_pga.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=4)

        self.card_sat = self._build_sat_card(self.cards_host)
        self.card_sat.grid(row=0, column=1, sticky="nsew", padx=10, pady=4)

        self.card_saavg = self._build_saavg_card(self.cards_host)
        self.card_saavg.grid(row=0, column=2, sticky="nsew", padx=(10, 0), pady=4)

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=3, column=0, sticky="ew", padx=28, pady=(0, 24))
        footer.grid_columnconfigure(0, weight=1)

        self.label_note = ctk.CTkLabel(
            footer,
            text=(
                "Note: for Sa(T) and Sa(avg), 'Calculated later' means the period will be estimated "
                "after plotting the capacity curve, using the linear section."
            ),
            font=("Inter", 12),
            justify="left",
            anchor="w",
            wraplength=780,
            text_color="#4b5563",
        )
        self.label_note.grid(row=0, column=0, sticky="w", pady=(0, 14))

        self.button_confirm = ctk.CTkButton(
            footer,
            text="Confirm",
            font=("Inter", 16, "bold"),
            height=48,
            command=self._on_confirm,
        )
        self.button_confirm.grid(row=1, column=0, sticky="ew")

    def _build_card_shell(self, parent, key: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, corner_radius=16, border_width=2)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(3, weight=1)
        self._bind_card_click(card, key)
        return card

    def _build_pga_card(self, parent) -> ctk.CTkFrame:
        card = self._build_card_shell(parent, "PGA")

        self.pga_badge = ctk.CTkLabel(card, text="", font=("Inter", 11, "bold"), anchor="w")
        self.pga_badge.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))

        self.pga_title = ctk.CTkLabel(card, text="PGA", font=("Inter", 24, "bold"), anchor="w")
        self.pga_title.grid(row=1, column=0, sticky="ew", padx=16)

        self.pga_desc = ctk.CTkLabel(
            card,
            text="Scale records using peak ground acceleration",
            font=("Inter", 13),
            justify="left",
            anchor="nw",
            wraplength=250,
        )
        self.pga_desc.grid(row=2, column=0, sticky="new", padx=16, pady=(10, 0))

        self.pga_footer = ctk.CTkLabel(
            card,
            text="No period controls are required for this workflow.",
            font=("Inter", 12),
            justify="left",
            anchor="sw",
            wraplength=250,
            text_color="#6b7280",
        )
        self.pga_footer.grid(row=3, column=0, sticky="sew", padx=16, pady=(16, 16))

        for widget in (self.pga_badge, self.pga_title, self.pga_desc, self.pga_footer):
            self._bind_card_click(widget, "PGA")
        return card

    def _build_sat_card(self, parent) -> ctk.CTkFrame:
        card = self._build_card_shell(parent, "Sa(T)")

        self.sat_badge = ctk.CTkLabel(card, text="", font=("Inter", 11, "bold"), anchor="w")
        self.sat_badge.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))

        self.sat_title = ctk.CTkLabel(card, text="Sa(T)", font=("Inter", 24, "bold"), anchor="w")
        self.sat_title.grid(row=1, column=0, sticky="ew", padx=16)

        self.sat_desc = ctk.CTkLabel(
            card,
            text="Scale records using spectral acceleration at a single period",
            font=("Inter", 13),
            justify="left",
            anchor="nw",
            wraplength=250,
        )
        self.sat_desc.grid(row=2, column=0, sticky="new", padx=16, pady=(10, 0))

        self.sat_controls = ctk.CTkFrame(card, fg_color="transparent")
        self.sat_controls.grid(row=3, column=0, sticky="sew", padx=16, pady=(16, 16))
        self.sat_controls.grid_columnconfigure(0, weight=1)

        self.sat_period_label = ctk.CTkLabel(
            self.sat_controls,
            text="Period source",
            font=("Inter", 13, "bold"),
            anchor="w",
        )
        self.sat_period_label.grid(row=0, column=0, sticky="w")

        self.sat_radio_calculated = ctk.CTkRadioButton(
            self.sat_controls,
            text="Calculated later",
            variable=self.sat_period_mode,
            value="calculated",
            command=self._refresh_cards,
        )
        self.sat_radio_calculated.grid(row=1, column=0, sticky="w", pady=(10, 4))

        self.sat_radio_specified = ctk.CTkRadioButton(
            self.sat_controls,
            text="Specified",
            variable=self.sat_period_mode,
            value="specified",
            command=self._refresh_cards,
        )
        self.sat_radio_specified.grid(row=2, column=0, sticky="w", pady=4)

        self.sat_t_frame = ctk.CTkFrame(self.sat_controls, fg_color="transparent")
        self.sat_t_frame.grid_columnconfigure(1, weight=1)
        self.sat_t_label = ctk.CTkLabel(self.sat_t_frame, text="T =", font=("Inter", 13, "bold"))
        self.sat_t_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.sat_t_entry = ctk.CTkEntry(self.sat_t_frame, textvariable=self.sat_period_value, width=120)
        self.sat_t_entry.grid(row=0, column=1, sticky="w")
        self.sat_t_unit = ctk.CTkLabel(self.sat_t_frame, text="s", font=("Inter", 13))
        self.sat_t_unit.grid(row=0, column=2, sticky="w", padx=(8, 0))

        for widget in (self.sat_badge, self.sat_title, self.sat_desc, self.sat_period_label):
            self._bind_card_click(widget, "Sa(T)")
        return card

    def _build_saavg_card(self, parent) -> ctk.CTkFrame:
        card = self._build_card_shell(parent, "Sa(avg)")

        self.saavg_badge = ctk.CTkLabel(card, text="", font=("Inter", 11, "bold"), anchor="w")
        self.saavg_badge.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))

        self.saavg_title = ctk.CTkLabel(card, text="Sa(avg)", font=("Inter", 24, "bold"), anchor="w")
        self.saavg_title.grid(row=1, column=0, sticky="ew", padx=16)

        self.saavg_desc = ctk.CTkLabel(
            card,
            text="Scale records using average spectral acceleration over a period range of 0.2T to 1.5T",
            font=("Inter", 13),
            justify="left",
            anchor="nw",
            wraplength=250,
        )
        self.saavg_desc.grid(row=2, column=0, sticky="new", padx=16, pady=(10, 0))

        self.saavg_controls = ctk.CTkFrame(card, fg_color="transparent")
        self.saavg_controls.grid(row=3, column=0, sticky="sew", padx=16, pady=(16, 16))
        self.saavg_controls.grid_columnconfigure(0, weight=1)

        self.saavg_period_label = ctk.CTkLabel(
            self.saavg_controls,
            text="Period source",
            font=("Inter", 13, "bold"),
            anchor="w",
        )
        self.saavg_period_label.grid(row=0, column=0, sticky="w")

        self.saavg_radio_calculated = ctk.CTkRadioButton(
            self.saavg_controls,
            text="Calculated later",
            variable=self.saavg_period_mode,
            value="calculated",
            command=self._refresh_cards,
        )
        self.saavg_radio_calculated.grid(row=1, column=0, sticky="w", pady=(10, 4))

        self.saavg_radio_specified = ctk.CTkRadioButton(
            self.saavg_controls,
            text="Specified",
            variable=self.saavg_period_mode,
            value="specified",
            command=self._refresh_cards,
        )
        self.saavg_radio_specified.grid(row=2, column=0, sticky="w", pady=4)

        self.saavg_t_frame = ctk.CTkFrame(self.saavg_controls, fg_color="transparent")
        self.saavg_t_frame.grid_columnconfigure(1, weight=1)
        self.saavg_t_label = ctk.CTkLabel(self.saavg_t_frame, text="T =", font=("Inter", 13, "bold"))
        self.saavg_t_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.saavg_t_entry = ctk.CTkEntry(self.saavg_t_frame, textvariable=self.saavg_period_value, width=120)
        self.saavg_t_entry.grid(row=0, column=1, sticky="w")
        self.saavg_t_unit = ctk.CTkLabel(self.saavg_t_frame, text="s", font=("Inter", 13))
        self.saavg_t_unit.grid(row=0, column=2, sticky="w", padx=(8, 0))

        for widget in (self.saavg_badge, self.saavg_title, self.saavg_desc, self.saavg_period_label):
            self._bind_card_click(widget, "Sa(avg)")
        return card

    def _bind_card_click(self, widget, key: str) -> None:
        widget.bind("<Button-1>", lambda _event, value=key: self._select_method(value))

    def _select_method(self, value: str) -> None:
        self.selected_method.set(value)
        self._refresh_cards()

    def _refresh_cards(self) -> None:
        selected = self.selected_method.get()
        self._style_card(self.card_pga, self.pga_badge, self.pga_title, selected == "PGA")
        self._style_card(self.card_sat, self.sat_badge, self.sat_title, selected == "Sa(T)")
        self._style_card(self.card_saavg, self.saavg_badge, self.saavg_title, selected == "Sa(avg)")

        if selected == "Sa(T)":
            self.sat_controls.grid()
        else:
            self.sat_controls.grid_remove()

        if selected == "Sa(avg)":
            self.saavg_controls.grid()
        else:
            self.saavg_controls.grid_remove()

        if self.sat_period_mode.get() == "specified":
            self.sat_t_frame.grid(row=3, column=0, sticky="w", pady=(10, 0))
        else:
            self.sat_t_frame.grid_remove()

        if self.saavg_period_mode.get() == "specified":
            self.saavg_t_frame.grid(row=3, column=0, sticky="w", pady=(10, 0))
        else:
            self.saavg_t_frame.grid_remove()

    def _style_card(self, card, badge, title, is_selected: bool) -> None:
        if is_selected:
            card.configure(border_color="#1f6aa5", fg_color="#f8fbff")
            badge.configure(text="SELECTED", text_color="#1f6aa5")
            title.configure(text_color="#1f6aa5")
        else:
            card.configure(border_color="#d1d5db", fg_color="#ffffff")
            badge.configure(text="", text_color="#6b7280")
            title.configure(text_color="#111827")

    def _on_confirm(self) -> None:
        try:
            self.result = self._collect_result()
        except ValueError as exc:
            messagebox.showerror("Invalid workflow setup", str(exc), parent=self)
            return

        logger.info("Workflow selected: %s", self.result)
        self.grab_release()
        self.destroy()

    def _collect_result(self) -> dict:
        method = self.selected_method.get()
        if method == "PGA":
            return {"im_method": "PGA", "period_mode": None, "period_value": None}

        if method == "Sa(T)":
            mode = self.sat_period_mode.get()
            period = self._parse_period(self.sat_period_value.get(), "Sa(T)") if mode == "specified" else None
            return {"im_method": "Sa(T)", "period_mode": mode, "period_value": period}

        if method == "Sa(avg)":
            mode = self.saavg_period_mode.get()
            period = self._parse_period(self.saavg_period_value.get(), "Sa(avg)") if mode == "specified" else None
            return {"im_method": "Sa(avg)", "period_mode": mode, "period_value": period}

        raise ValueError("Please select one workflow option before continuing.")

    @staticmethod
    def _parse_period(raw_value: str, label: str) -> float:
        text = raw_value.strip()
        if not text:
            raise ValueError(f"Please provide T in seconds for {label}.")
        try:
            period = float(text)
        except ValueError as exc:
            raise ValueError(f"T for {label} must be a valid number.") from exc
        if period <= 0:
            raise ValueError(f"T for {label} must be greater than 0.")
        return period

    def _on_close(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()
