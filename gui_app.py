#!/usr/bin/env python3
"""
GUI Version of DXF BOQ Extractor
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
from main_app import main as process_files
import sys
from io import StringIO

class DXFBOQApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DXF BOQ Extractor")
        self.root.geometry("900x700")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="📐 DXF BOQ Extractor", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input folder
        ttk.Label(main_frame, text="Input Folder:").grid(row=1, column=0, sticky=tk.W)
        self.input_var = tk.StringVar(value="input")
        ttk.Entry(main_frame, textvariable=self.input_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input).grid(row=1, column=2)
        
        # Output folder
        ttk.Label(main_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, pady=10)
        self.output_var = tk.StringVar(value="output")
        ttk.Entry(main_frame, textvariable=self.output_var, width=40).grid(row=2, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=2, column=2)
        
        # Units selection
        ttk.Label(main_frame, text="Drawing Units:").grid(row=3, column=0, sticky=tk.W, pady=10)
        self.units_var = tk.StringVar(value="m")
        units_frame = ttk.Frame(main_frame)
        units_frame.grid(row=3, column=1, columnspan=2, sticky=tk.W)
        
        for unit in ["m", "mm", "cm"]:
            ttk.Radiobutton(units_frame, text=unit, variable=self.units_var, value=unit).pack(side=tk.LEFT, padx=10)
        
        # Project name
        ttk.Label(main_frame, text="Project Name:").grid(row=4, column=0, sticky=tk.W, pady=10)
        self.project_var = tk.StringVar(value="My Engineering Project")
        ttk.Entry(main_frame, textvariable=self.project_var, width=40).grid(row=4, column=1, columnspan=2, sticky=tk.W)
        
        # Process button
        ttk.Button(main_frame, text="🚀 Process DXF Files", 
                  command=self.process_files, 
                  style="Accent.TButton").grid(row=5, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Output text area
        ttk.Label(main_frame, text="Processing Output:").grid(row=7, column=0, sticky=tk.W, pady=10)
        
        # Create scrolled text widget
        self.output_text = scrolledtext.ScrolledText(main_frame, height=20, width=80)
        self.output_text.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(8, weight=1)
        
        # Add some style
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))
        
    def browse_input(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_var.set(folder)
            
    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_var.set(folder)
            
    def process_files(self):
        # Clear previous output
        self.output_text.delete(1.0, tk.END)
        
        # Validate folders
        input_folder = Path(self.input_var.get())
        if not input_folder.exists():
            messagebox.showerror("Error", f"Input folder not found:\n{input_folder}")
            return
        
        # Start processing in separate thread
        self.progress.start()
        thread = threading.Thread(target=self._process_files_thread)
        thread.daemon = True
        thread.start()
        
    def _process_files_thread(self):
        try:
            # Redirect stdout to capture output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            # Simulate command line arguments
            class Args:
                def __init__(self):
                    self.input = self.input_var.get()
                    self.output = self.output_var.get()
                    self.units = self.units_var.get()
                    self.project = self.project_var.get()
                    self.verbose = False
                    self.list_only = False
            
            args = Args()
            
            # Call the main processing function
            from main_app import main as process_main
            
            # We need to modify sys.argv for the argparser
            import sys as sys_module
            sys_module.argv = ['main_app.py', 
                              '--input', args.input,
                              '--output', args.output,
                              '--units', args.units,
                              '--project', args.project]
            
            process_main()
            
            # Get captured output
            output = sys.stdout.getvalue()
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Update GUI
            self.root.after(0, self._update_output, output)
            self.root.after(0, self.progress.stop)
            
            # Show success message
            output_folder = Path(self.output_var.get())
            if output_folder.exists():
                files = list(output_folder.glob("*"))
                if files:
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Success", 
                        f"Processing complete!\n\n"
                        f"Generated {len(files)} files in:\n{output_folder}"
                    ))
            
        except Exception as e:
            sys.stdout = old_stdout
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n{str(e)}"))
            
    def _update_output(self, text):
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)

def main():
    root = tk.Tk()
    app = DXFBOQApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()