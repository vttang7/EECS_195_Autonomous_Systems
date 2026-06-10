#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import curses

class TurtleTeleop:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('my_teleop_node')
        self.pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.twist = Twist()
        
        # Movement parameters
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.5  # rad/s

    def send_command(self, linear, angular):
        """Publish movement command"""
        self.twist.linear.x = linear
        self.twist.angular.z = angular
        self.pub.publish(self.twist)

    def run(self):
        # Initialize curses for keyboard input
        stdscr = curses.initscr()
        curses.cbreak()
        stdscr.keypad(True)
        stdscr.nodelay(True)
        
        try:
            while not rospy.is_shutdown():
                key = stdscr.getch()
                
                # Handle key presses
                if key == ord('w'):  # Move forward
                    self.send_command(self.linear_speed, 0.0)
                elif key == ord('s'):  # Move backward
                    self.send_command(-self.linear_speed, 0.0)
                elif key == ord('a'):  # Rotate counterclockwise
                    self.send_command(0.0, self.angular_speed)
                elif key == ord('d'):  # Rotate clockwise
                    self.send_command(0.0, -self.angular_speed)
                elif key == ord('q'):
                    # Quit the program
                    self.send_command(0.0, 0.0)
                    break
                
                
        finally:
            # Clean up curses
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.endwin()

if __name__ == '__main__':
    try:
        teleop = TurtleTeleop()
        teleop.run()
    except rospy.ROSInterruptException:
        pass