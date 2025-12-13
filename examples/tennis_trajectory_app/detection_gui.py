"""GUI integration module that connects the model upload popup with the detection engine."""

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import logging

# Add the core module to the path
sys.path.insert(0, str(Path(__file__).parent))

from core import DetectionEngine, DetectionMode, ProcessingMode
from gui_popup import show_model_upload_popup


class DetectionEngineGUI:
    """Main GUI application for the tennis trajectory detection engine."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.engine = None
        self.current_mode = DetectionMode.BALL
        self.setup_window()
        self.setup_widgets()
        
        # Check if models are already configured
        self.check_existing_models()
    
    def setup_window(self):
        """Set up the main window."""
        self.root.title("🎾 Tennis Trajectory Detection Engine")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Configure window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_widgets(self):
        """Set up the main GUI widgets."""
        # Main frame with padding
        self.main_frame = tk.Frame(self.root, padx=20, pady=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            self.main_frame,
            text="🎾 Tennis Trajectory Detection Engine",
            font=("Arial", 18, "bold"),
            fg="#2E8B57"
        )
        title_label.pack(pady=(0, 20))
        
        # Mode selection frame
        mode_frame = tk.LabelFrame(self.main_frame, text="Detection Mode", font=("Arial", 12, "bold"))
        mode_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Mode selection buttons
        button_frame = tk.Frame(mode_frame)
        button_frame.pack(pady=10)
        
        self.mode_vars = {}
        modes = [("Ball Detection", DetectionMode.BALL), 
                 ("Racket Detection", DetectionMode.RACKET),
                 ("Player Detection", DetectionMode.PLAYER)]
        
        for i, (text, mode) in enumerate(modes):
            var = tk.BooleanVar()
            btn = tk.Radiobutton(
                button_frame,
                text=text,
                variable=var,
                value=True if mode == self.current_mode else False,
                command=lambda m=mode: self.on_mode_change(m),
                font=("Arial", 10),
                indicatoron=False,
                width=15,
                relief=tk.FLAT,
                bd=2
            )
            btn.grid(row=0, column=i, padx=5, pady=5)
            self.mode_vars[mode] = var
            if mode == self.current_mode:
                var.set(True)
        
        # Status frame
        self.status_frame = tk.LabelFrame(self.main_frame, text="Engine Status", font=("Arial", 12, "bold"))
        self.status_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Status content
        self.status_label = tk.Label(
            self.status_frame,
            text="Checking engine status...",
            font=("Arial", 10),
            anchor="w"
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=10)
        
        # Model management frame
        model_frame = tk.LabelFrame(self.main_frame, text="Model Management", font=("Arial", 12, "bold"))
        model_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Model management buttons
        model_button_frame = tk.Frame(model_frame)
        model_button_frame.pack(pady=10)
        
        self.configure_models_btn = tk.Button(
            model_button_frame,
            text="📁 Configure Models",
            command=self.configure_models,
            relief=tk.FLAT,
            bg="#007BFF",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20,
            pady=5
        )
        self.configure_models_btn.pack(side=tk.LEFT, padx=5)
        
        self.refresh_status_btn = tk.Button(
            model_button_frame,
            text="🔄 Refresh Status",
            command=self.refresh_status,
            relief=tk.FLAT,
            bg="#28A745",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20,
            pady=5
        )
        self.refresh_status_btn.pack(side=tk.LEFT, padx=5)
        
        # Processing frame
        processing_frame = tk.LabelFrame(self.main_frame, text="Frame Processing", font=("Arial", 12, "bold"))
        processing_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Processing buttons
        processing_button_frame = tk.Frame(processing_frame)
        processing_button_frame.pack(pady=10)
        
        self.process_image_btn = tk.Button(
            processing_button_frame,
            text="🖼️ Process Image",
            command=self.process_image,
            relief=tk.FLAT,
            bg="#17A2B8",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20,
            pady=5,
            state="disabled"
        )
        self.process_image_btn.pack(side=tk.LEFT, padx=5)
        
        self.process_video_btn = tk.Button(
            processing_button_frame,
            text="🎥 Process Video",
            command=self.process_video,
            relief=tk.FLAT,
            bg="#6F42C1",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20,
            pady=5,
            state="disabled"
        )
        self.process_video_btn.pack(side=tk.LEFT, padx=5)
        
        # Info text area
        self.info_text = tk.Text(
            processing_frame,
            height=8,
            width=70,
            wrap=tk.WORD,
            font=("Consolas", 9),
            relief=tk.FLAT,
            bg="#F8F9FA",
            state=tk.DISABLED
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Action buttons frame
        action_frame = tk.Frame(self.main_frame)
        action_frame.pack(fill=tk.X)
        
        self.show_demo_btn = tk.Button(
            action_frame,
            text="🎯 Run Demo",
            command=self.run_demo,
            relief=tk.FLAT,
            bg="#FD7E14",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=30,
            pady=8
        )
        self.show_demo_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.exit_btn = tk.Button(
            action_frame,
            text="❌ Exit",
            command=self.on_closing,
            relief=tk.FLAT,
            bg="#DC3545",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=30,
            pady=8
        )
        self.exit_btn.pack(side=tk.RIGHT)
    
    def check_existing_models(self):
        """Check if models are already configured."""
        try:
            self.engine = DetectionEngine()
            self.refresh_status()
            
            # Check if we have active models for the current mode
            active_model = self.engine.model_manager.get_active_model(self.current_mode)
            if active_model:
                self.enable_processing_buttons()
                self.log_info(f"✓ Active model for {self.current_mode.value}: {active_model}")
            else:
                self.log_info("⚠ No active model found. Click 'Configure Models' to set up YOLOv8n models.")
                
        except Exception as e:
            self.log_info(f"❌ Error checking models: {e}")
    
    def on_mode_change(self, mode):
        """Handle detection mode change."""
        self.current_mode = mode
        
        # Update radio button states
        for m, var in self.mode_vars.items():
            var.set(m == mode)
        
        # Check if we have an active model for this mode
        if self.engine:
            active_model = self.engine.model_manager.get_active_model(mode)
            if active_model:
                self.enable_processing_buttons()
                self.log_info(f"✓ Switched to {mode.value} mode. Active model: {active_model}")
            else:
                self.disable_processing_buttons()
                self.log_info(f"⚠ No active model for {mode.value} mode. Configure models first.")
    
    def configure_models(self):
        """Show the model configuration popup."""
        choice, model_paths = show_model_upload_popup(self.root)
        
        if choice == "models":
            self.setup_real_models(model_paths)
        elif choice == "placeholders":
            self.setup_placeholder_models()
        else:
            self.log_info("Model configuration cancelled.")
    
    def setup_real_models(self, model_paths):
        """Set up real YOLOv8n models."""
        try:
            if not self.engine:
                self.engine = DetectionEngine()
            
            success_count = 0
            for mode, path in model_paths.items():
                if path:
                    mode_enum = DetectionMode(mode)
                    model_name = f"gui_{mode}_{Path(path).stem}"
                    
                    if self.engine.model_manager.register_model(model_name, path, mode_enum):
                        if self.engine.model_manager.set_active_model(mode_enum, model_name):
                            success_count += 1
                            self.log_info(f"✓ Registered and activated {model_name} for {mode} mode")
            
            if success_count > 0:
                self.refresh_status()
                self.enable_processing_buttons()
                messagebox.showinfo("Success", f"Successfully configured {success_count} models!")
            else:
                messagebox.showerror("Error", "Failed to configure any models.")
                
        except Exception as e:
            error_msg = f"Error configuring models: {e}"
            self.log_info(f"❌ {error_msg}")
            messagebox.showerror("Error", error_msg)
    
    def setup_placeholder_models(self):
        """Set up placeholder models for testing."""
        try:
            if not self.engine:
                self.engine = DetectionEngine()
            
            for mode in DetectionMode:
                model_name = f"placeholder_{mode.value}"
                model_path = self.create_placeholder_model(model_name)
                
                if self.engine.model_manager.register_model(model_name, model_path, mode):
                    self.engine.model_manager.set_active_model(mode, model_name)
                    self.log_info(f"✓ Set up placeholder model for {mode.value} mode")
            
            self.refresh_status()
            self.enable_processing_buttons()
            messagebox.showinfo("Success", "Placeholder models configured for testing!")
            
        except Exception as e:
            error_msg = f"Error setting up placeholders: {e}"
            self.log_info(f"❌ {error_msg}")
            messagebox.showerror("Error", error_msg)
    
    def create_placeholder_model(self, model_name: str) -> str:
        """Create a placeholder model file."""
        placeholder_path = Path(f"{model_name}.pt")
        with open(placeholder_path, 'wb') as f:
            header = b"TENNIS_YOLOv8_GUI_PLACEHOLDER\x00"
            f.write(header)
            padding = b'\x00' * (1024 * 1024)  # 1MB placeholder
            f.write(padding)
        return str(placeholder_path)
    
    def refresh_status(self):
        """Refresh the engine status display."""
        try:
            if not self.engine:
                self.status_label.config(text="Engine not initialized")
                return
            
            status = self.engine.get_status()
            
            # Format status text
            status_text = f"Models: {status['model_manager_status']['total_models']} | "
            status_text += f"Modes: {', '.join(status['available_modes'])} | "
            status_text += f"CUDA: {'Yes' if status['model_manager_status']['available_devices']['cuda'] else 'No'} | "
            status_text += f"Active: {self.current_mode.value}"
            
            # Color coding
            if status['model_manager_status']['total_models'] > 0:
                status_color = "#28A745"  # Green
            else:
                status_color = "#DC3545"  # Red
            
            self.status_label.config(text=status_text, fg=status_color)
            
        except Exception as e:
            self.status_label.config(text=f"Error: {e}", fg="#DC3545")
    
    def enable_processing_buttons(self):
        """Enable processing buttons."""
        self.process_image_btn.config(state="normal")
        self.process_video_btn.config(state="normal")
    
    def disable_processing_buttons(self):
        """Disable processing buttons."""
        self.process_image_btn.config(state="disabled")
        self.process_video_btn.config(state="disabled")
    
    def log_info(self, message: str):
        """Log information to the info text area."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, f"{message}\n")
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)
    
    def process_image(self):
        """Process an image file."""
        from tkinter import filedialog
        
        filepath = filedialog.askopenfilename(
            title="Select Image to Process",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            # Import here to avoid issues if opencv is not installed
            import cv2
            
            # Load image
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Could not load image")
            
            self.log_info(f"🖼️ Processing image: {Path(filepath).name}")
            
            # Process with current mode
            if self.engine:
                results = self.engine.process(image, self.current_mode)
                
                if results:
                    self.log_info(f"✓ Detected {len(results)} objects:")
                    for i, obj in enumerate(results, 1):
                        self.log_info(f"  {i}. {obj.class_name} - Confidence: {obj.confidence:.3f}")
                        self.log_info(f"     Position: {obj.centroid}, Size: {obj.width}x{obj.height}")
                else:
                    self.log_info("⚠ No objects detected")
            else:
                self.log_info("❌ Engine not available")
                
        except ImportError:
            self.log_info("❌ OpenCV not available. Install with: pip install opencv-python")
        except Exception as e:
            self.log_info(f"❌ Error processing image: {e}")
    
    def process_video(self):
        """Process a video file."""
        from tkinter import filedialog
        
        filepath = filedialog.askopenfilename(
            title="Select Video to Process",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            # Import here to avoid issues if opencv is not installed
            import cv2
            
            # Open video
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                raise ValueError("Could not open video")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.log_info(f"🎥 Processing video: {Path(filepath).name}")
            self.log_info(f"   FPS: {fps:.1f}, Frames: {frame_count}")
            
            # Process first few frames as demo
            process_frames = min(10, frame_count)
            
            if self.engine:
                for i in range(process_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = self.engine.process(frame, self.current_mode)
                    if results:
                        self.log_info(f"  Frame {i+1}: {len(results)} objects detected")
            
            self.log_info("✓ Video processing completed")
            
            cap.release()
            
        except ImportError:
            self.log_info("❌ OpenCV not available. Install with: pip install opencv-python")
        except Exception as e:
            self.log_info(f"❌ Error processing video: {e}")
    
    def run_demo(self):
        """Run the demonstration."""
        try:
            self.log_info("🎯 Starting demonstration...")
            
            # Import and run demo
            import demo
            import io
            import contextlib
            
            # Capture demo output
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                demo.main()
            
            demo_output = f.getvalue()
            self.log_info("✓ Demo completed successfully")
            
            # Show demo summary
            messagebox.showinfo("Demo Completed", 
                              "The demonstration ran successfully!\n\n"
                              "Check the info panel above for details.")
            
        except Exception as e:
            self.log_info(f"❌ Demo failed: {e}")
            messagebox.showerror("Demo Error", f"Demo failed with error:\n{e}")
    
    def on_closing(self):
        """Handle window closing."""
        # Ask for confirmation
        if messagebox.askokcancel("Quit", "Do you want to quit the Tennis Trajectory Detection Engine?"):
            try:
                # Cleanup engine if needed
                if self.engine:
                    self.engine.model_manager.unload_all_models()
            except:
                pass
            
            self.root.destroy()
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point for the GUI application."""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create and run the GUI
        app = DetectionEngineGUI()
        app.run()
        
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        # Fallback to showing popup directly
        try:
            from gui_popup import show_model_upload_popup
            show_model_upload_popup()
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")


if __name__ == "__main__":
    main()