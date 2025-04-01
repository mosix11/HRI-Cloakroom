import subprocess
import time

def main():
    print("Enter your transcript messages (Ctrl+C to exit).")

    try:
        while True:
            # Take user input for the transcript
            transcript = input("Enter transcript message: ")

            # Get the current timestamp in ISO format
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

            # Define the start and end times for the transcription (you can adjust this)
            start_time = 0.0
            end_time = 2.3

            # Construct the ROS2 command
            command = [
                "ros2", "topic", "pub", "--once", "/audio/transcription", 
                "my_msgs/msg/Transcript", 
                f"{{transcript: '{transcript}', timestamp: '{timestamp}', start: {start_time}, end: {end_time}}}"
            ]

            # Run the command in the terminal
            subprocess.run(command)

            print(f"Published transcript: {transcript} at {timestamp}")

    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
