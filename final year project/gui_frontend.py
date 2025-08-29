import tkinter as tk
from tkinter import messagebox
import subprocess
import os

def run_script(script_name):
    try:
        if os.path.exists(script_name):
            subprocess.Popen(["py", script_name], shell=True)
        else:
            messagebox.showerror("Error", f"{script_name} not found!")
    except Exception as e:
        messagebox.showerror("Execution Failed", str(e))

root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("500x400")
root.configure(bg="#e9f0f7")

tk.Label(root, text="Face Recognition Attendance System", font=("Helvetica", 16, "bold"), bg="#e9f0f7").pack(pady=20)

tk.Button(root, text="1. Collect Face Data", font=("Arial", 12), width=30,
          command=lambda: run_script("face_data_collection.py")).pack(pady=10)

tk.Button(root, text="2. Train Model", font=("Arial", 12), width=30,
          command=lambda: run_script("train_model.py")).pack(pady=10)

tk.Button(root, text="3. Run Real-Time Recognition", font=("Arial", 12), width=30,
          command=lambda: run_script("main.py")).pack(pady=10)

tk.Button(root, text="4. Test Face Detection", font=("Arial", 12), width=30,
          command=lambda: run_script("face_recognition_module.py")).pack(pady=10)

tk.Button(root, text="Exit", font=("Arial", 12), width=30, command=root.quit).pack(pady=20)

root.mainloop()
