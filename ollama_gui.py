#!/usr/bin/env python3
"""
Ollama GUI Client - Tkinter interface for Ollama text generation
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import json
import requests
from typing import Optional
import sys
from datetime import datetime

OLLAMA_HOST = "http://localhost:11434"  # Change if your server listens elsewhere


class OllamaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ollama GUI Client")
        self.root.geometry("900x700")
        
        # Variables
        self.model_var = tk.StringVar(value="gpt-oss:20b")
        self.gpu_var = tk.IntVar(value=99)  # 99 = full GPU
        self.stream_var = tk.BooleanVar(value=True)
        self.is_generating = False
        
        # Available models
        self.available_models = []
        
        # Setup UI
        self.setup_ui()
        
        # Load available models
        self.load_models()
        
    def setup_ui(self):
        """Setup the user interface"""
        
        # Top control panel
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model selection
        ttk.Label(control_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, width=30)
        self.model_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # GPU/CPU toggle
        ttk.Label(control_frame, text="Processor:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        gpu_frame = ttk.Frame(control_frame)
        gpu_frame.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        self.gpu_radio_gpu = ttk.Radiobutton(gpu_frame, text="GPU (Full)", variable=self.gpu_var, value=99)
        self.gpu_radio_gpu.pack(side=tk.LEFT)
        self.gpu_radio_cpu = ttk.Radiobutton(gpu_frame, text="CPU Only", variable=self.gpu_var, value=0)
        self.gpu_radio_cpu.pack(side=tk.LEFT)
        
        # Stream toggle (inline with Processor)
        self.stream_check = ttk.Checkbutton(control_frame, text="Stream Output", 
                                           variable=self.stream_var)
        self.stream_check.grid(row=0, column=4, sticky=tk.W, padx=15, pady=5)
        
        # Stats display - Fixed frame with labels
        stats_frame = ttk.LabelFrame(self.root, text="Current Model Stats", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create a grid layout for stats
        self.stats_model_label = ttk.Label(stats_frame, text="Model: -", font=("Arial", 9, "bold"))
        self.stats_model_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        self.stats_url_label = ttk.Label(stats_frame, text=f"URL: {OLLAMA_HOST}", font=("Arial", 9, "bold"))
        self.stats_url_label.grid(row=0, column=2, columnspan=2, sticky=tk.W, padx=20, pady=2)
        
        # Timing stats
        ttk.Label(stats_frame, text="Timing:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.stats_total_label = ttk.Label(stats_frame, text="Total: -", font=("Arial", 9, "bold"))
        self.stats_total_label.grid(row=1, column=1, sticky=tk.W, padx=10)
        self.stats_load_label = ttk.Label(stats_frame, text="Load: -", font=("Arial", 9, "bold"))
        self.stats_load_label.grid(row=1, column=2, sticky=tk.W, padx=10)
        self.stats_prompt_eval_label = ttk.Label(stats_frame, text="Prompt Eval: -", font=("Arial", 9, "bold"))
        self.stats_prompt_eval_label.grid(row=1, column=3, sticky=tk.W, padx=10)
        
        self.stats_gen_label = ttk.Label(stats_frame, text="Generation: -", font=("Arial", 9, "bold"))
        self.stats_gen_label.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=10, pady=2)
        
        # Token stats
        ttk.Label(stats_frame, text="Tokens:", font=("Arial", 9, "bold")).grid(row=3, column=0, sticky=tk.W, pady=2)
        self.stats_prompt_tokens_label = ttk.Label(stats_frame, text="Prompt: -", font=("Arial", 9, "bold"))
        self.stats_prompt_tokens_label.grid(row=3, column=1, sticky=tk.W, padx=10)
        self.stats_gen_tokens_label = ttk.Label(stats_frame, text="Generated: -", font=("Arial", 9, "bold"))
        self.stats_gen_tokens_label.grid(row=3, column=2, sticky=tk.W, padx=10)
        self.stats_total_tokens_label = ttk.Label(stats_frame, text="Total: -", font=("Arial", 9, "bold"))
        self.stats_total_tokens_label.grid(row=3, column=3, sticky=tk.W, padx=10)
        
        # Performance stats
        ttk.Label(stats_frame, text="Performance:", font=("Arial", 9, "bold")).grid(row=4, column=0, sticky=tk.W, pady=2)
        self.stats_prompt_speed_label = ttk.Label(stats_frame, text="Prompt Eval: -", font=("Arial", 9, "bold"))
        self.stats_prompt_speed_label.grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=10)
        self.stats_gen_speed_label = ttk.Label(stats_frame, text="Generation: -", font=("Arial", 9, "bold"))
        self.stats_gen_speed_label.grid(row=4, column=3, sticky=tk.W, padx=10)
        
        # Completion info
        self.stats_completion_label = ttk.Label(stats_frame, text="Status: No generation yet", 
                                                font=("Arial", 9, "bold"), foreground="gray")
        self.stats_completion_label.grid(row=5, column=0, columnspan=4, sticky=tk.W, pady=2)
        
        # Prompt input
        prompt_frame = ttk.LabelFrame(self.root, text="Prompt", padding=10)
        prompt_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.prompt_entry = scrolledtext.ScrolledText(prompt_frame, height=4, wrap=tk.WORD)
        self.prompt_entry.pack(fill=tk.BOTH, expand=True)
        self.prompt_entry.insert("1.0", "Tell me a joke about programming")
        
        # Generate button
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.generate_btn = ttk.Button(button_frame, text="Generate", command=self.generate)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_generation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="Clear Output", command=self.clear_output)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(button_frame, text="Ready", foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Output display
        output_frame = ttk.LabelFrame(self.root, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, 
                                                     font=("Arial", 10))
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
    def load_models(self):
        """Load available models from Ollama"""
        try:
            resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            data = resp.json()
            self.available_models = [model["name"] for model in data.get("models", [])]
            self.model_combo["values"] = self.available_models
            
            if self.available_models and self.model_var.get() not in self.available_models:
                self.model_var.set(self.available_models[0])
                
        except Exception as e:
            self.output_text.insert(tk.END, f"Error loading models: {e}\n")
            self.model_combo["values"] = ["gpt-oss:20b", "qwen3-coder"]
    
    def update_status(self, message, color="black"):
        """Update status label"""
        self.status_label.config(text=message, foreground=color)
    
    def update_stats(self, data):
        """Update the stats display with generation information"""
        if not data:
            return
        
        # Model info
        if "model" in data:
            self.stats_model_label.config(text=f"Model: {data['model']}")
        
        # Timing
        total_duration = data.get("total_duration", 0) / 1e9
        load_duration = data.get("load_duration", 0) / 1e9
        prompt_eval_duration = data.get("prompt_eval_duration", 0) / 1e9
        eval_duration = data.get("eval_duration", 0) / 1e9
        
        self.stats_total_label.config(text=f"Total: {total_duration:.3f}s")
        self.stats_load_label.config(text=f"Load: {load_duration:.3f}s")
        self.stats_prompt_eval_label.config(text=f"Prompt Eval: {prompt_eval_duration:.3f}s")
        self.stats_gen_label.config(text=f"Generation: {eval_duration:.3f}s")
        
        # Tokens
        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)
        
        self.stats_prompt_tokens_label.config(text=f"Prompt: {prompt_eval_count}")
        self.stats_gen_tokens_label.config(text=f"Generated: {eval_count}")
        self.stats_total_tokens_label.config(text=f"Total: {prompt_eval_count + eval_count}")
        
        # Performance
        if prompt_eval_duration > 0 and prompt_eval_count > 0:
            prompt_speed = prompt_eval_count / prompt_eval_duration
            self.stats_prompt_speed_label.config(text=f"Prompt Eval: {prompt_speed:.1f} tok/s")
        else:
            self.stats_prompt_speed_label.config(text="Prompt Eval: -")
        
        if eval_duration > 0 and eval_count > 0:
            gen_speed = eval_count / eval_duration
            self.stats_gen_speed_label.config(text=f"Generation: {gen_speed:.1f} tok/s")
        else:
            self.stats_gen_speed_label.config(text="Generation: -")
        
        # Done reason
        if "done_reason" in data:
            self.stats_completion_label.config(
                text=f"Status: Completed ({data['done_reason']})",
                foreground="green"
            )
    
    def generate(self):
        """Start generation in a separate thread"""
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        if not prompt:
            self.update_status("Please enter a prompt", "red")
            return
        
        if self.is_generating:
            self.update_status("Already generating...", "orange")
            return
        
        # Disable controls
        self.is_generating = True
        self.generate_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_status("Generating...", "blue")
        
        # Clear output
        self.output_text.insert(tk.END, f"\n{'='*70}\n")
        self.output_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Prompt: {prompt}\n")
        self.output_text.insert(tk.END, f"{'='*70}\n")
        self.output_text.see(tk.END)
        
        # Start generation thread
        thread = threading.Thread(target=self._generate_thread, args=(prompt,))
        thread.daemon = True
        thread.start()
    
    def _generate_thread(self, prompt):
        """Thread function for generation"""
        try:
            model = self.model_var.get()
            num_gpu = self.gpu_var.get()
            stream = self.stream_var.get()
            
            url = f"{OLLAMA_HOST}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
            }
            
            if num_gpu is not None:
                payload["options"] = {"num_gpu": num_gpu}
            
            if stream:
                self._generate_stream(url, payload)
            else:
                self._generate_non_stream(url, payload)
                
        except Exception as e:
            self.root.after(0, self._append_output, f"\n[ERROR] {e}\n")
            self.root.after(0, self.update_status, "Error", "red")
        finally:
            self.root.after(0, self._generation_complete)
    
    def _generate_stream(self, url, payload):
        """Handle streaming generation"""
        last_data = None
        
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not self.is_generating:
                    break
                    
                if not line:
                    continue
                
                data = json.loads(line.decode("utf-8"))
                chunk = data.get("response", "")
                
                if chunk:
                    self.root.after(0, self._append_output, chunk)
                
                if data.get("done", False):
                    last_data = data
        
        self.root.after(0, self._append_output, "\n")
        
        # Always show stats
        if last_data:
            self.root.after(0, self.update_stats, last_data)
    
    def _generate_non_stream(self, url, payload):
        """Handle non-streaming generation"""
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        
        response_text = data.get("response", "")
        self.root.after(0, self._append_output, response_text)
        self.root.after(0, self._append_output, "\n")
        
        # Always show stats
        self.root.after(0, self.update_stats, data)
    
    def _append_output(self, text):
        """Append text to output (called from main thread)"""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
    
    def _generation_complete(self):
        """Called when generation is complete"""
        self.is_generating = False
        self.generate_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status("Ready", "green")
    
    def stop_generation(self):
        """Stop the current generation"""
        self.is_generating = False
        self.update_status("Stopping...", "orange")
    
    def clear_output(self):
        """Clear the output text"""
        self.output_text.delete("1.0", tk.END)


def main():
    root = tk.Tk()
    app = OllamaGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

