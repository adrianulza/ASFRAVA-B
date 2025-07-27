import queue

import customtkinter as ctk


class ProgressTracker:
    def __init__(self, master):
        # state
        self._total: int = 1
        self._done:  int = 0             
        self._after_id: int | None = None  
        self._q: queue.Queue[int] = queue.Queue()

        # container frame
        self._frame = ctk.CTkFrame(master, fg_color="transparent")
        self._frame.grid(row=4, column=1, columnspan=3, sticky="wes", padx=(0, 0), pady=(0, 0))

        # configure three columns: static label, bar, status label
        self._frame.columnconfigure(0, weight=0)
        self._frame.columnconfigure(1, weight=1)
        self._frame.columnconfigure(2, weight=0)

        # Static label on the left
        self._static_label = ctk.CTkLabel(self._frame, text="Progress bar", font=("Inter bold", 10, "bold"), anchor="w")
        self._static_label.grid(row=0, column=0, sticky="w", padx=(5, 5), pady=(0, 0))

        # The progress bar in the middle
        self._bar = ctk.CTkProgressBar(self._frame, height=4, progress_color="green")
        self._bar.grid(row=0, column=1, sticky="we", padx=(5, 15), pady=(0, 0))
        self._bar.set(0)

        # Dynamic status label on the right
        self._status_label = ctk.CTkLabel(self._frame, text="Status: Idle", font=("Inter", 10, "normal"), anchor="e")
        self._status_label.grid(row=0, column=2, sticky="e", padx=(5, 110), pady=(0, 0))

        self._master = master

    # -------------------------------------------------------------------- #
    # public API                                                          #
    # -------------------------------------------------------------------- #
    def start(self, total_steps: int):
        """
        Call once just before you launch the worker‑thread.
        Resets everything so the bar starts fresh.
        """
        # reset state
        self._total = max(total_steps, 1)
        self._done = 0
        self._bar.set(0)
        self._status_label.configure(text="Idle")

        # stop any previous after–loop
        if self._after_id is not None:
            self._master.after_cancel(self._after_id)

        # kick off a new polling cycle
        self._after_id = self._master.after(100, self._poll)

    def tick(self):
        """Thread‑safe – put one unit of completed work onto the queue."""
        self._q.put(1)

    # -------------------------------------------------------------------- #
    # force‑complete                                                      #
    # -------------------------------------------------------------------- #
    def finish(self):
        """Snap the bar to 100 % and show Done immediately."""
        self._bar.set(1.0)
        self._status_label.configure(text="Status: Done ✓")
        self._done = self._total

    # -------------------------------------------------------------------- #
    # internal                                                            #
    # -------------------------------------------------------------------- #
    def _poll(self):
        """Drain the queue and refresh the GUI."""
        moved = 0
        try:
            while True:
                self._q.get_nowait()
                moved += 1
        except queue.Empty:
            pass

        if moved:
            self._done += moved 

            fraction = self._done / self._total
            self._bar.set(fraction)
            self._status_label.configure(text=f"{self._done:,}/{self._total:,}")

        if self._done < self._total:

            self._after_id = self._master.after(100, self._poll)
        else:

            self._after_id = None
            self._bar.set(1.0)
            self._status_label.configure(text="Status: Done ✓")
