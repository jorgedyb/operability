#!/usr/bin/env python3
"""
Operability App GUI: select hull type, dimensions, and wave direction (in degrees) to lookup RAO results and plot magnitudes per DOF.
Only CSV-based RAO outputs are supported.
"""
import os
import re
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def lookup_raos(input_dir, hull, length, beam, draft):
    """
    Load RAO CSV for a given (length, beam, draft) case.
    """
    def fmt(x):
        try:
            return str(int(float(x))) if float(x).is_integer() else str(x)
        except Exception:
            return str(x)

    Hull, L, B, T = fmt(hull), fmt(length), fmt(beam), fmt(draft)
    fname = os.path.join(input_dir, f"{Hull}_L{L}_B{B}_T{T}_rao.csv")
    if os.path.exists(fname):
        return pd.read_csv(fname)
    raise FileNotFoundError(f"No RAO CSV for L={length}, B={beam}, T={draft} in {input_dir}")


def to_magnitude(x):
    """Convert a value to its magnitude if complex-like."""
    try:
        return abs(complex(x))
    except Exception:
        try:
            return abs(float(x))
        except Exception:
            return np.nan

class OperabilityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Operability App')
        self.geometry('800x700')
        # Configure grid to expand plot
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.wave_map = {}
        self.create_widgets()

    def create_widgets(self):
        # Control panel frame
        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky='ew')
        for i in range(3):
            frm.grid_columnconfigure(i, weight=1)

        # Row 0: Hull Type
        ttk.Label(frm, text='Hull Type:').grid(row=0, column=0, sticky='w')
        self.hull_var = tk.StringVar(value='barge')
        hull_cb = ttk.Combobox(frm, textvariable=self.hull_var,
                               values=['barge','OSV'], state='readonly')
        hull_cb.grid(row=0, column=1, sticky='ew')

        # Rows 1-3: Dimensions
        dims = ['length', 'beam', 'draft']
        self.dim_cbs = {}
        for idx, name in enumerate(dims, start=1):
            ttk.Label(frm, text=name.capitalize() + ':').grid(row=idx, column=0, sticky='w')
            cb = ttk.Combobox(frm, state='readonly')
            cb.grid(row=idx, column=1, sticky='ew')
            cb.bind('<<ComboboxSelected>>', lambda e: self.load_wave_options())
            self.dim_cbs[name] = cb

        # Wave Direction
        ttk.Label(frm, text='Wave Direction (°):').grid(row=4, column=0, sticky='w')
        self.wave_cb = ttk.Combobox(frm, state='disabled')
        self.wave_cb.grid(row=4, column=1, sticky='ew')

        # RAO Directory
        ttk.Label(frm, text='RAO Directory:').grid(row=5, column=0, sticky='w')
        self.dir_entry = ttk.Entry(frm)
        self.dir_entry.insert(0, 'rao_outputs_fine')
        self.dir_entry.grid(row=5, column=1, sticky='ew')
        browse_btn = ttk.Button(frm, text='Browse', command=self.browse_dir)
        browse_btn.grid(row=5, column=2, sticky='w')

        # Lookup & Plot button
        lookup_btn = ttk.Button(frm, text='Lookup & Plot', command=self.on_lookup)
        lookup_btn.grid(row=6, column=1, pady=5)

        # Plot area
        plot_frame = ttk.Frame(self)
        plot_frame.grid(row=1, column=0, sticky='nsew')
        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(6,4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Initialize dropdowns
        self.load_options()

    def browse_dir(self):
        d = filedialog.askdirectory(initialdir=self.dir_entry.get())
        if d:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, d)
            self.load_options()

    def load_options(self):
        input_dir = self.dir_entry.get()
        pattern = re.compile(r'barge_L(?P<L>[^_]+)_B(?P<B>[^_]+)_T(?P<T>[^_]+)_rao\.csv$')
        mapping = {'length':'L', 'beam':'B', 'draft':'T'}
        vals = {key:set() for key in mapping}
        if os.path.isdir(input_dir):
            for fname in os.listdir(input_dir):
                m = pattern.match(fname)
                if m:
                    for key, grp in mapping.items():
                        vals[key].add(m.group(grp))
        for key, cb in self.dim_cbs.items():
            cb['values'] = sorted(vals[key], key=lambda x: float(x))
            cb.set('')
        self.wave_cb.set('')
        self.wave_cb['state'] = 'disabled'
        self.ax.clear()
        self.canvas.draw()

    def load_wave_options(self):
        l = self.dim_cbs['length'].get()
        b = self.dim_cbs['beam'].get()
        d = self.dim_cbs['draft'].get()
        if not all((l, b, d)):
            return
        try:
            df = lookup_raos(self.dir_entry.get(),'OSV', l, b, d)
            dirs = df['wave_direction'].unique().tolist()
            degs = []
            self.wave_map.clear()
            for w in sorted(dirs, key=float):
                deg_str = f"{math.degrees(w):.1f}"
                degs.append(deg_str)
                self.wave_map[deg_str] = w
            self.wave_cb['values'] = degs
            if degs:
                self.wave_cb.set(degs[0])
            self.wave_cb['state'] = 'readonly'
        except FileNotFoundError:
            pass

    def on_lookup(self):
        l = self.dim_cbs['length'].get()
        b = self.dim_cbs['beam'].get()
        d = self.dim_cbs['draft'].get()
        w = self.wave_cb.get()
        dir_path = self.dir_entry.get()
        if not all((l, b, d, w)):
            messagebox.showerror('Error', 'Select all inputs')
            return
        try:
            df = lookup_raos(dir_path,'barge', l, b, d)
        except FileNotFoundError as e:
            messagebox.showerror('Error', str(e))
            return
        rad = self.wave_map[w]
        df = df[df['wave_direction'] == float(rad)].copy()
        df['RAO'] = df['RAO'].apply(to_magnitude)
        pivot = df.pivot(index='omega', columns='radiating_dof', values='RAO')
        self.ax.clear()
        for dof in pivot.columns:
            self.ax.plot(pivot.index, pivot[dof], label=dof)
        self.ax.set(xlabel='Omega [rad/s]', ylabel='RAO Magnitude',
                    title=f'RAO vs Omega @ {w}°')
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        self.canvas.draw()

if __name__ == '__main__':
    OperabilityApp().mainloop()
