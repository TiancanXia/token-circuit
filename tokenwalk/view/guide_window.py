"""Scrollable document window. Content and game execution stay outside this view."""

import tkinter as tk
from tkinter import ttk
from ..content.guide import TOPICS, document, topic_for
from ..content.attention_guide import FORMULA_TITLES
from ..i18n import Locale
from .painting import BG, TEXT, MUTED, GOLD
from .formulas import attention_images


class GuideWindow:
    def __init__(
        self,
        root,
        config,
        language,
        guided,
        cache_enabled,
        phase,
        topic,
        cn,
        mono,
        on_close,
    ):
        self.context = (config, language, guided, cache_enabled, phase)
        self.on_close = on_close
        self.locale = Locale(language)
        self.window = tk.Toplevel(root)
        self.window.title(self.locale.text("guide.title"))
        self.window.geometry("820x700")
        self.window.minsize(620, 460)
        self.window.configure(bg=BG)
        self.window.transient(root)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda event: self.close())
        style = ttk.Style(self.window)
        style.theme_use("clam")
        style.configure(
            "Handbook.TCombobox",
            background="#253e51",
            fieldbackground="#253e51",
            foreground=TEXT,
            arrowcolor=GOLD,
            bordercolor="#314651",
        )
        style.map(
            "Handbook.TCombobox",
            fieldbackground=[("readonly", "#253e51")],
            foreground=[("readonly", TEXT)],
        )
        style.configure(
            "Handbook.Vertical.TScrollbar",
            background="#355361",
            troughcolor=BG,
            arrowcolor=GOLD,
            bordercolor=BG,
        )
        self.window.option_add("*TCombobox*Listbox.background", "#253e51")
        self.window.option_add("*TCombobox*Listbox.foreground", TEXT)
        header = tk.Frame(self.window, bg=BG, padx=28, pady=20)
        header.pack(fill="x")
        tk.Label(
            header, text=self.locale.text("guide.title"), font=(cn, -25), bg=BG, fg=TEXT
        ).pack(side="left")
        tk.Button(
            header,
            text=self.locale.text("guide.close"),
            command=self.close,
            font=(cn, -14),
            bg="#253e51",
            fg=TEXT,
            activebackground="#344c48",
            activeforeground=TEXT,
            relief="flat",
            padx=14,
            pady=7,
        ).pack(side="right")
        self.topics = tuple(TOPICS)
        self.selector = ttk.Combobox(
            self.window,
            state="readonly",
            values=[TOPICS[t].title for t in self.topics],
            font=(cn, -16),
            style="Handbook.TCombobox",
        )
        self.selector.pack(fill="x", padx=28, pady=(0, 15))
        self.selector.bind("<<ComboboxSelected>>", self.select)
        # Reserve the footer before the expanding document, including at the
        # minimum window size. Text's natural requested height is much larger.
        tk.Label(
            self.window,
            text=self.locale.text("guide.footer"),
            font=(cn, -13),
            bg=BG,
            fg=MUTED,
            pady=16,
        ).pack(side="bottom", fill="x")
        body = tk.Frame(self.window, bg=BG)
        body.pack(fill="both", expand=True, padx=28)
        scroll = ttk.Scrollbar(
            body, orient="vertical", style="Handbook.Vertical.TScrollbar"
        )
        scroll.pack(side="right", fill="y")
        self.text = tk.Text(
            body,
            wrap="word",
            bg="#172b3a",
            fg=TEXT,
            font=(cn, -17),
            relief="flat",
            borderwidth=0,
            padx=24,
            pady=20,
            spacing1=4,
            spacing3=12,
            height=1,
            width=1,
            insertbackground=GOLD,
            yscrollcommand=scroll.set,
        )
        self.text.pack(fill="both", expand=True)
        scroll.configure(command=self.text.yview)
        self.text.tag_configure("title", font=(cn, -28), foreground=GOLD, spacing3=20)
        self.text.tag_configure(
            "heading", font=(cn, -19, "bold"), foreground=GOLD, spacing1=18, spacing3=8
        )
        self.text.tag_configure(
            "formula", font=(mono, -16), foreground="#a3cfdd", spacing1=4, spacing3=8
        )
        self.formula_images = []
        self.show_topic(topic_for(topic))
        self.window.grab_set()
        self.text.focus_set()

    def select(self, event=None):
        self.show_topic(self.topics[self.selector.current()])

    def show_topic(self, topic):
        self.topic = topic
        self.selector.current(self.topics.index(topic))
        page = document(topic, *self.context)
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.formula_images = []
        self.text.insert("end", page.title + "\n", "title")
        for index, (heading, body) in enumerate(page.sections):
            self.text.insert("end", heading + "\n", "heading")
            self.text.insert(
                "end",
                body + "\n",
                "formula" if index == 2 and topic != "attention" else (),
            )
            if topic == "attention" and index == 3:
                self.formula_images = attention_images(self.window)
                for titles, image in zip(FORMULA_TITLES, self.formula_images):
                    self.text.insert(
                        "end", titles[self.locale.language == "en"] + "\n", "heading"
                    )
                    self.text.image_create("end", image=image, padx=0, pady=8)
                    self.text.insert("end", "\n")
                if not self.formula_images:
                    self.text.insert("end", TOPICS[topic].calculation + "\n", "formula")
        self.text.configure(state="disabled")
        self.text.yview_moveto(0.0)

    def close(self):
        if self.window.winfo_exists():
            self.window.grab_release()
            self.window.destroy()
            self.on_close()
