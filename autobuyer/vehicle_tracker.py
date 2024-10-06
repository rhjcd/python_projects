import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import statistics
import os
import numpy as np
import pyperclip

class VehicleTracker:
    def __init__(self, master):
        self.master = master
        self.master.title("Vehicle Search Tracker")
        self.master.geometry("800x600")

        self.vehicles = []
        self.search_params = {
            'min_year': 1999,
            'max_year': 2014,
            'min_price': 2000,
            'max_price': 7000,
            'make': 'Toyota'
        }
        self.load_data()

        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill='both')

        self.add_frame = ttk.Frame(self.notebook)
        self.view_frame = ttk.Frame(self.notebook)
        self.analytics_frame = ttk.Frame(self.notebook)
        self.search_params_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.add_frame, text='Add Vehicle')
        self.notebook.add(self.view_frame, text='View Vehicles')
        self.notebook.add(self.analytics_frame, text='Analytics')
        self.notebook.add(self.search_params_frame, text='Search Parameters')

        self.create_add_vehicle_widgets()
        self.create_view_vehicles_widgets()
        self.create_analytics_widgets()
        self.create_search_params_widgets()

    def create_add_vehicle_widgets(self):
        fields = ['Year', 'Model', 'Price', 'Mileage', 'Location', 'Source', 'URL']
        self.entries = {}

        for i, field in enumerate(fields):
            label = ttk.Label(self.add_frame, text=field)
            label.grid(row=i, column=0, padx=5, pady=5, sticky='e')
            entry = ttk.Entry(self.add_frame, width=50)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[field.lower()] = entry

            paste_button = ttk.Button(self.add_frame, text="Paste", command=lambda f=field.lower(): self.paste_from_clipboard(f))
            paste_button.grid(row=i, column=2, padx=5, pady=5)

        notes_label = ttk.Label(self.add_frame, text="Notes")
        notes_label.grid(row=len(fields), column=0, padx=5, pady=5, sticky='ne')
        self.notes_text = tk.Text(self.add_frame, width=50, height=5)
        self.notes_text.grid(row=len(fields), column=1, padx=5, pady=5)

        add_button = ttk.Button(self.add_frame, text="Add Vehicle", command=self.add_vehicle)
        add_button.grid(row=len(fields)+1, column=1, pady=10)

        paste_all_button = ttk.Button(self.add_frame, text="Paste All Fields", command=self.paste_all_fields)
        paste_all_button.grid(row=len(fields)+1, column=2, pady=10)

    def create_view_vehicles_widgets(self):
        self.tree = ttk.Treeview(self.view_frame, columns=('Year', 'Model', 'Price', 'Mileage', 'Location', 'Source'), show='headings')
        self.tree.pack(expand=True, fill='both')

        for col in self.tree['columns']:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        self.update_treeview()

        button_frame = ttk.Frame(self.view_frame)
        button_frame.pack(fill='x', pady=10)

        update_button = ttk.Button(button_frame, text="Update", command=self.update_vehicle)
        update_button.pack(side='left', padx=5)

        delete_button = ttk.Button(button_frame, text="Delete", command=self.delete_vehicle)
        delete_button.pack(side='left', padx=5)

        export_button = ttk.Button(button_frame, text="Export", command=self.export_data)
        export_button.pack(side='left', padx=5)

    def create_analytics_widgets(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.analytics_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill='both')

        self.update_analytics()

    def create_search_params_widgets(self):
        fields = ['min_year', 'max_year', 'min_price', 'max_price', 'make']
        self.param_entries = {}

        for i, field in enumerate(fields):
            label = ttk.Label(self.search_params_frame, text=field.replace('_', ' ').title())
            label.grid(row=i, column=0, padx=5, pady=5, sticky='e')
            entry = ttk.Entry(self.search_params_frame, width=20)
            entry.insert(0, str(self.search_params[field]))
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.param_entries[field] = entry

        save_button = ttk.Button(self.search_params_frame, text="Save Parameters", command=self.save_search_params)
        save_button.grid(row=len(fields), column=1, pady=10)

    def add_vehicle(self):
        vehicle = {field: entry.get() for field, entry in self.entries.items()}
        vehicle['notes'] = self.notes_text.get("1.0", tk.END).strip()

        if not self.validate_input(vehicle):
            return

        self.vehicles.append(vehicle)
        self.save_data()
        self.update_treeview()
        self.update_analytics()
        messagebox.showinfo("Success", "Vehicle added successfully!")
        self.clear_entries()

    def validate_input(self, vehicle):
        try:
            year = int(vehicle['year'])
            price = float(vehicle['price'])
            if not (self.search_params['min_year'] <= year <= self.search_params['max_year']):
                raise ValueError(f"Year must be between {self.search_params['min_year']} and {self.search_params['max_year']}")
            if not (self.search_params['min_price'] <= price <= self.search_params['max_price']):
                raise ValueError(f"Price must be between ${self.search_params['min_price']} and ${self.search_params['max_price']}")
            if vehicle['model'].split()[0].lower() != self.search_params['make'].lower():
                raise ValueError(f"Make must be {self.search_params['make']}")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return False
        return True

    def update_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for vehicle in self.vehicles:
            self.tree.insert('', 'end', values=(vehicle['year'], vehicle['model'], vehicle['price'],
                                                vehicle['mileage'], vehicle['location'], vehicle['source']))

    def update_analytics(self):
        if not self.vehicles:
            return

        self.ax1.clear()
        self.ax2.clear()

        years = [int(v['year']) for v in self.vehicles]
        prices = [float(v['price']) for v in self.vehicles]

        self.ax1.bar(years, prices)
        self.ax1.set_xlabel('Year')
        self.ax1.set_ylabel('Price ($)')
        self.ax1.set_title('Price by Year')

        sources = [v['source'] for v in self.vehicles]
        source_counts = {s: sources.count(s) for s in set(sources)}
        self.ax2.pie(source_counts.values(), labels=source_counts.keys(), autopct='%1.1f%%')
        self.ax2.set_title('Listings by Source')

        self.canvas.draw()

    def update_vehicle(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a vehicle to update.")
            return
        
        # Implementation for updating a vehicle

    def delete_vehicle(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a vehicle to delete.")
            return
        
        index = self.tree.index(selected_item)
        del self.vehicles[index]
        self.save_data()
        self.update_treeview()
        self.update_analytics()

    def export_data(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv")
        if file_path:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.vehicles[0].keys())
                writer.writeheader()
                writer.writerows(self.vehicles)
            messagebox.showinfo("Success", f"Data exported to {file_path}")

    def load_data(self):
        if os.path.exists('vehicles.csv'):
            with open('vehicles.csv', 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                self.vehicles = list(reader)

    def save_data(self):
        with open('vehicles.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.vehicles[0].keys())
            writer.writeheader()
            writer.writerows(self.vehicles)

    def clear_entries(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.notes_text.delete("1.0", tk.END)

    def paste_from_clipboard(self, field):
        clipboard_content = pyperclip.paste()
        self.entries[field].delete(0, tk.END)
        self.entries[field].insert(0, clipboard_content)

    def paste_all_fields(self):
        clipboard_content = pyperclip.paste().split('\n')
        fields = list(self.entries.keys())
        for i, content in enumerate(clipboard_content):
            if i < len(fields):
                self.entries[fields[i]].delete(0, tk.END)
                self.entries[fields[i]].insert(0, content.strip())

    def save_search_params(self):
        try:
            for key, entry in self.param_entries.items():
                if key in ['min_year', 'max_year', 'min_price', 'max_price']:
                    self.search_params[key] = int(entry.get())
                else:
                    self.search_params[key] = entry.get()
            messagebox.showinfo("Success", "Search parameters updated successfully!")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for year and price fields.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleTracker(root)
    root.mainloop()
