"""GUI popup utilities for model upload instructions."""

import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import sys
import os
from pathlib import Path


class ModelUploadPopup:
    """Popup window for YOLOv8n model upload instructions."""
    
    def __init__(self, parent=None):
        self.parent = parent or tk.Tk()
        self.result = None
        self.model_paths = {}
        
        # Hide parent window if provided
        if parent:
            parent.withdraw()
            
        self.setup_window()
    
    def setup_window(self):
        """Set up the popup window."""
        self.window = tk.Toplevel(self.parent)
        self.window.title("🎾 YOLOv8n Model Setup Required")
        self.window.geometry("700x600")
        self.window.resizable(True, True)
        
        # Center the window
        self.window.transient(self.parent)
        self.window.grab_set()
        
        # Make window close properly
        self.window.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create and arrange widgets in the popup."""
        # Main frame
        main_frame = tk.Frame(self.window, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="🎾 Tennis Trajectory Detection Engine",
            font=("Arial", 16, "bold"),
            fg="#2E8B57"
        )
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle_label = tk.Label(
            main_frame,
            text="YOLOv8n Models Required for Detection",
            font=("Arial", 12),
            fg="#666666"
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Instructions text
        instructions_text = """
To use the tennis trajectory detection engine, you need to provide YOLOv8n models:

📦 REQUIRED MODELS:
• Ball Detection Model (.pt file)
• Racket Detection Model (.pt file)  
• Player Detection Model (.pt file)

🔗 WHERE TO GET MODELS:
• Download YOLOv8n pretrained models from Ultralytics
• Train custom models on tennis-specific datasets
• Use models from tennis tracking research papers

📁 MODEL REQUIREMENTS:
• Format: YOLOv8n .pt files (Ultralytics format)
• Optimized for tennis scene analysis
• File size: Typically 5-20MB per model
• Classes should include tennis_ball, racket, person

⚙️ MODEL SETUP:
1. Use the "Browse" buttons to select your model files
2. Or use the CLI: python cli.py --add-model ball --yolo-model tennis_ball.pt
3. Models will be stored in the 'models/' directory

🔧 TESTING WITHOUT MODELS:
• You can still test the engine with placeholder models
• All functionality will work, but no real detections will occur
        """
        
        # Create scrolled text for instructions
        text_widget = ScrolledText(
            main_frame,
            height=15,
            width=80,
            wrap=tk.WORD,
            font=("Arial", 10),
            relief=tk.FLAT,
            bg="#F8F9FA"
        )
        text_widget.pack(pady=(0, 20), fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, instructions_text.strip())
        text_widget.config(state=tk.DISABLED)
        
        # Model selection frame
        model_frame = tk.LabelFrame(main_frame, text="Model Selection (Optional)", font=("Arial", 10, "bold"))
        model_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.model_vars = {}
        model_frame_inner = tk.Frame(model_frame, padx=10, pady=10)
        model_frame_inner.pack(fill=tk.X)
        
        # Model selection rows
        models = [
            ("Ball Detection", "ball"),
            ("Racket Detection", "racket"),
            ("Player Detection", "player")
        ]
        
        for display_name, mode in models:
            row_frame = tk.Frame(model_frame_inner)
            row_frame.pack(fill=tk.X, pady=2)
            
            # Label
            label = tk.Label(row_frame, text=display_name, width=20, anchor="w")
            label.pack(side=tk.LEFT)
            
            # File path entry
            var = tk.StringVar()
            entry = tk.Entry(row_frame, textvariable=var, width=40)
            entry.pack(side=tk.LEFT, padx=(10, 5))
            
            # Browse button
            browse_btn = tk.Button(
                row_frame,
                text="Browse...",
                command=lambda m=mode, v=var: self.browse_model(m, v),
                relief=tk.FLAT,
                bg="#E9ECEF"
            )
            browse_btn.pack(side=tk.LEFT)
            
            self.model_vars[mode] = var
        
        # Action buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create buttons
        self.setup_btn = tk.Button(
            button_frame,
            text="📁 Browse All Models",
            command=self.browse_all_models,
            relief=tk.FLAT,
            bg="#007BFF",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20
        )
        self.setup_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.use_placeholders_btn = tk.Button(
            button_frame,
            text="🧪 Use Placeholder Models",
            command=self.use_placeholder_models,
            relief=tk.FLAT,
            bg="#28A745",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20
        )
        self.use_placeholders_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.exit_btn = tk.Button(
            button_frame,
            text="❌ Exit",
            command=self.on_cancel,
            relief=tk.FLAT,
            bg="#DC3545",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20
        )
        self.exit_btn.pack(side=tk.RIGHT)
        
        # Information frame
        info_frame = tk.LabelFrame(main_frame, text="Quick Info", font=("Arial", 9, "bold"))
        info_frame.pack(fill=tk.X)
        
        info_text = tk.Label(
            info_frame,
            text="💡 Tip: You can always add models later using the CLI interface",
            font=("Arial", 9),
            fg="#6C757D"
        )
        info_text.pack(pady=5)
        
        # Set focus to browse button
        self.setup_btn.focus_set()
    
    def browse_model(self, mode, var):
        """Browse for a specific model file."""
        filetypes = [
            ("YOLOv8n models", "*.pt"),
            ("PyTorch models", "*.pth"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title=f"Select {mode.title()} Detection Model",
            filetypes=filetypes
        )
        
        if filename:
            var.set(filename)
    
    def browse_all_models(self):
        """Browse for all model files at once."""
        # Create a directory selection dialog
        model_dir = filedialog.askdirectory(
            title="Select directory containing YOLOv8n model files"
        )
        
        if model_dir:
            model_dir = Path(model_dir)
            
            # Auto-detect model files
            model_files = {
                "ball": None,
                "racket": None,
                "player": None
            }
            
            # Look for common model file patterns
            for file_path in model_dir.glob("*.pt"):
                filename = file_path.name.lower()
                if "ball" in filename:
                    model_files["ball"] = str(file_path)
                elif "racket" in filename or "paddle" in filename:
                    model_files["racket"] = str(file_path)
                elif "player" in filename or "person" in filename or "player" in filename:
                    model_files["player"] = str(file_path)
            
            # Update the entry fields
            for mode, file_path in model_files.items():
                if file_path and os.path.exists(file_path):
                    self.model_vars[mode].set(file_path)
            
            # Show info message
            matched = sum(1 for v in model_files.values() if v is not None)
            messagebox.showinfo(
                "Models Found",
                f"Auto-detected {matched} model files from the selected directory.\n\n"
                f"Please verify the file paths and adjust if needed."
            )
    
    def use_placeholder_models(self):
        """Use placeholder models for testing."""
        messagebox.showinfo(
            "Placeholder Models",
            "The engine will create placeholder models for testing.\n\n"
            "✓ All functionality will work\n"
            "✗ No real detections will occur\n\n"
            "You can add real models later via CLI or this interface."
        )
        self.result = "placeholders"
        self.on_ok()
    
    def on_cancel(self):
        """Handle cancel button click."""
        self.result = "cancel"
        self.window.destroy()
        
        # Show parent window if it was hidden
        if self.parent and hasattr(self.parent, 'deiconify'):
            self.parent.deiconify()
    
    def on_ok(self):
        """Handle OK button click."""
        self.result = "models"
        self.model_paths = {
            mode: var.get() 
            for mode, var in self.model_vars.items() 
            if var.get().strip()
        }
        
        if not self.model_paths:
            messagebox.showwarning(
                "No Models Selected",
                "Please select at least one model file or choose 'Use Placeholder Models'."
            )
            return
        
        # Validate selected models
        invalid_models = []
        for mode, path in self.model_paths.items():
            if not os.path.exists(path):
                invalid_models.append(f"{mode}: {path}")
        
        if invalid_models:
            messagebox.showerror(
                "Invalid Model Paths",
                "The following model files do not exist:\n" + "\n".join(invalid_models)
            )
            return
        
        self.window.destroy()
        
        # Show parent window if it was hidden
        if self.parent and hasattr(self.parent, 'deiconify'):
            self.parent.deiconify()
    
    def show(self):
        """Show the popup and return the result."""
        self.window.wait_window()
        return self.result, self.model_paths


def show_model_upload_popup(parent=None):
    """
    Show the model upload popup and return user choice.
    
    Args:
        parent: Parent Tkinter window (optional)
        
    Returns:
        tuple: (choice, model_paths)
            choice: "models", "placeholders", or "cancel"
            model_paths: dict of mode -> file_path if choice == "models"
    """
    popup = ModelUploadPopup(parent)
    return popup.show()


def create_basic_tk_window():
    """Create a basic Tkinter window for testing."""
    root = tk.Tk()
    root.title("Tennis Trajectory Detection Engine")
    root.geometry("400x300")
    
    # Main content
    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    title_label = tk.Label(
        main_frame,
        text="🎾 Tennis Trajectory Detection Engine",
        font=("Arial", 14, "bold")
    )
    title_label.pack(pady=(0, 20))
    
    def on_start():
        # Show model upload popup
        choice, model_paths = show_model_upload_popup(root)
        
        if choice == "models":
            messagebox.showinfo("Models Selected", f"Selected models:\n{model_paths}")
            # Here you would integrate with the actual detection engine
            # engine = DetectionEngine()
            # engine.load_models(model_paths)
        elif choice == "placeholders":
            messagebox.showinfo("Using Placeholders", "Using placeholder models for testing.")
            # engine = DetectionEngine()
            # engine.use_placeholder_models()
        else:
            messagebox.showinfo("Cancelled", "Model setup cancelled.")
    
    start_btn = tk.Button(
        main_frame,
        text="🎯 Start Detection Engine",
        command=on_start,
        relief=tk.FLAT,
        bg="#007BFF",
        fg="white",
        font=("Arial", 12, "bold"),
        padx=30,
        pady=10
    )
    start_btn.pack(pady=20)
    
    status_label = tk.Label(
        main_frame,
        text="Click 'Start' to configure YOLOv8n models",
        font=("Arial", 10),
        fg="#6C757D"
    )
    status_label.pack(pady=(10, 0))
    
    return root


if __name__ == "__main__":
    # Test the popup
    root = create_basic_tk_window()
    root.mainloop()