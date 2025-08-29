from datetime import datetime

def mark_attendance(name):
    now = datetime.now()
    time_string = now.strftime('%H:%M:%S')
    date_string = now.strftime('%Y-%m-%d')
    filename = f"attendance_{date_string}.csv"

    try:
        with open(filename, 'r+') as f:
            lines = f.readlines()
            names_logged = [line.split(',')[0] for line in lines]

            if name not in names_logged:
                f.write(f"{name},{time_string}\n")
    except FileNotFoundError:
        with open(filename, 'w') as f:
            f.write("Name,Time\n")
            f.write(f"{name},{time_string}\n")
