from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)
import io
import sys
import threading
import time

from pyfirmata import Arduino, util

import paho.mqtt.client as mqtt

import picamera
import io

handler_method_prefix = 'rmsg_'

class mqtt_node(object):
    def __init__(self, Subscriptions=[], Blocking=False):
        self.blocking_mode = Blocking
        self.subscriptions = Subscriptions
        self.broker_host = "localhost"
        self.broker_host = "192.168.8.101"
        self.broker_port = 1883
        self.broker_timeout = 60

    def Connect(self):
        self.mqttc = mqtt.Client()
        # Assign event callbacks
        self.mqttc.on_message = self.on_message
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_publish = self.on_publish
        self.mqttc.on_subscribe = self.on_subscribe
        # Connect
        self.mqttc.connect(self.broker_host, self.broker_port, self.broker_timeout)
        if self.blocking_mode:
            self.mqttc.loop_forever()
        else:
            self.mqttc.loop_start()

    def Disconnect(self):
        if self.blocking_mode:
            pass
        else:
            self.mqttc.loop_stop(force=False)

    def on_connect(self, client, userdata, flags, rc):
        print("rc: " + str(rc))
        for this_topic in self.subscriptions:
            handler_name = handler_method_prefix + this_topic
            if not hasattr(self, handler_name):
                print("No message handler for topic '%s'" % (this_topic))
            self.mqttc.subscribe(this_topic, 0)
        print(self.subscriptions)

    def on_message(self, client, userdata, message):
        print(message.topic + " " + str(message.qos) + " " + str(message.payload))
        handler_name = handler_method_prefix + message.topic
        handler_method = getattr(self, handler_name)
        handler_method(message.payload.decode("utf-8"))

    def on_publish(self, client, userdata, mid):
        print("mid: " + str(mid))

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    def on_log(self, client, userdata, level, buf):
        print(buf)

def Test_Mqtt_Node():
    n = mqtt_node(Subscriptions=['test'], Blocking=True)
    n.Connect()

class vehicle(object):
    """
        This class isolates low level hardware functions so that helmsman is vehicle
        agnostic. Right now it is hardwired for my initial robot. Later on it will
        either be subclassed or specilaized with a configuration file.

        For now, speed variables are actual Arduino Servo values. Eventually
        we want them to use actual speed mm/sec and map that to whatever
        control values are needed for the vehicle.
    """
    def __init__(self):
        self.board = Arduino('/dev/ttyUSB0')
        self.motor = self.board.get_pin('d:9:s')
        self.mot_offset = 90
        self.mot_goal = 0		# This is the pulse we are ramping towards
        self.mot_jump = 10			# This is the minimum speed to start moving from stop
        self.mot_ramp = 0			# Current ramping increment
        self.mot_last_pulse = 0
        self.mot_last_pulse_commit = 0
        self.motor.write(self.mot_offset)	# Stop motor if on
        self.steering = self.board.get_pin('d:10:s')
        self.st_straight = 90
        # speed in mm/second - depends on vehicle and battery condition
        self.speed_crawl_forward = 8		# minimum start moving speed
        self.speed_increment = 1		# a reasonable quantity for "go a bit faster"
        self.speed_max = 13411			# 30mph / 13.4112 meters/second
        self.steering_increment	= 10		# degrees of casual steering adjustment
        self.steering_max = 60			# 60 degrees left or right

    def ConvertSpeedToPulseParameter(self, speed):
        # for now, speed is just arduino servo increment value.
        # Degree of servo turn, but cetnered at 0 instead of 90.
        return speed

    def Motor(self, speed_goal):
        # This sends commands to the hardware motor controller (ESC or H-Bridge).
        # This handles ramping if not handled by hardware motor controller.
        # This only considers forward motion right now.
        # This is fragile. Need to soften states to avoid race conditions.
        pulse_goal = self.ConvertSpeedToPulseParameter(speed_goal)
        if pulse_goal != self.mot_goal:
            # the goal has changed, need to reset ramping variables
            if pulse_goal == 0:
                # we want to stop
                self.mot_last_pulse = pulse_goal
                self.mot_goal = pulse_goal
                self.mot_ramp = 0
            elif (pulse_goal != 0) and (self.mot_last_pulse == 0):
                # we are starting to move
                if abs(pulse_goal) > self.mot_jump:
                    # we are starting fast, so just do it
                    self.mot_last_pulse = pulse_goal
                    self.mot_goal = pulse_goal
                    self.mot_ramp = 0
                else:
                    # we are starting slow, need to make an initial jump
                    self.mot_goal = pulse_goal
                    if pulse_goal > 0:
                      self.mot_last_pulse = self.mot_jump
                      self.mot_ramp = -1
                    else:
                      self.mot_last_pulse = -self.mot_jump
                      self.mot_ramp = +1
            else:
                # this is speed change while moving
                self.mot_last_pulse = pulse_goal
                self.mot_goal = pulse_goal
                self.mot_ramp = 0
        else:
            # No change in goal, keep ramping toward that
            if self.mot_ramp != 0:
                self.mot_last_pulse += self.mot_ramp
                if self.mot_last_pulse == self.mot_goal:
                    self.mot_ramp = 0
        self.motor.write(self.mot_offset + self.mot_last_pulse)
        if self.mot_last_pulse_commit != self.mot_last_pulse:
           print('Motor: ', self.mot_last_pulse)
           self.mot_last_pulse_commit = self.mot_last_pulse

    def Steering(self, direction):
         self.steering.write(90+direction)

def cameraman(helmsman):
    # This will run in its own thread.
    # Touch helmsman as little as possible to avoid thread glitches.
    with picamera.PiCamera() as camera:
        camera.iso = 800
        camera.shutter_speed = 10000		# microseconds, 1000 = 1ms
        camera.vflip = True
        # Camera warm-up time
        time.sleep(2)
        prev_mode = 'x'				# x is invalid, forces startup in single mode
        while True:
            if helmsman.camera_mode != prev_mode:
                if prev_mode == 's':
                    # switching to run mode
                    prev_mode = 'r'
                    sleep_interval = 0.1
                    run_ct = 0
                else:
                    # switching to single mode
                    prev_mode = 's'
                    sleep_interval = 1
            if prev_mode == 's':
                picfn = 'temp/single.jpg'
            else: 
                run_ct += 1
                picfn = 'temp/R%d_%d.jpg' % (run_ct, time.clock())
            #my_stream = io.BytesIO()
            #camera.capture(my_stream, 'jpeg')
            if (prev_mode == 'r') or (helmsman.camera_snap == True):
              camera.capture(picfn, 'jpeg')
              helmsman.camera_last_fn = picfn
              if prev_mode == 's':
                  # There is a potential race condition here where we miss the second of two
                  # closely timed requests. We will still have taken a photo very recently
                  # and published that. That shoud be good enough.
                  helmsman.camera_snap = False
            time.sleep(sleep_interval)

class helmsman(mqtt_node):
    def __init__(self):
        super().__init__(Subscriptions=('set_speed', 'steer', 'take_pic'), Blocking=False)
        self.v = vehicle()
        self.camera_mode = 's'		# set by helmsman, s=single, r=run
        self.camera_snap = False	# set by helmsman, cleared by cameraman
        self.camera_last_fn = None	# set by camerman
        self.speed_goal = 0		# (int) mm/sec
        self.steering_goal = 0		# (int) degrees (0 = straigh, neg is degrees left, pos is degrees right)
        self.camera = threading.Thread(target=cameraman, args=(self,))
        self.camera.start()

    def rmsg_take_pic(self, msg):
        # should we verify mode and report if a problem?
        self.camera_snap = True

    def rmsg_set_speed(self, msg):
        self.GetGoalSpeed(msg)
        print(self.speed_goal)

    def rmsg_steer(self, msg):
        self.GetGoalSteering(msg)
        print(self.steering_goal)

    def Loop(self):
      while True:
          self.Process()
          time.sleep(0.1)

    def Process(self):
        self.GetGoal()
        self.GetActuals()
        self.AdjustDriving()

    def GetGoal(self):
        pass

    def GetActuals(self):
        pass

    def AdjustDriving(self):
        # This is just a place holder
        if self.speed_goal == 0:
            self.camera_mode = 's'
        else:
            self.camera_mode = 'r'
            #time.sleep(2)
        self.v.Motor(self.speed_goal)
        self.v.Steering(self.steering_goal)

    def GetGoalSpeed(self, speed_request):
        if speed_request in '+=':
          speed_goal = self.speed_goal + self.v.speed_increment
        elif speed_request == '-':
          speed_goal = self.speed_goal - self.v.speed_increment
        elif speed_request == 'f':			# move forward slowly
          speed_goal = self.v.speed_crawl_forward
        elif speed_request == 's':			# stop moving
          speed_goal = 0
        else:
          try:
            speed_goal = int(speed_request)
          except:
            print("Bad Input '%s'" %(speed_request))
            speed_goal = self.speed_goal
        if abs(speed_goal) > self.v.speed_max:
            if speed_goal > 0:
                self.speed_goal = +self.v.speed_max
            else:
                self.speed_goal = -self.v.speed_max
        else:
            self.speed_goal = speed_goal

    def GetGoalSteering(self, steering_request):
        if steering_request == 's':
            steering_goal = 0
        elif steering_request == '+l':
            steering_goal = self.steering_goal - self.v.steering_increment
        elif steering_request == '+r':
            steering_goal = self.steering_goal + self.v.steering_increment
        else:
            try:
                steering_goal = int(steering_request)
            except:
                print("Bad Steering Input '%s'" % (steering_request))
                steering_goal = sself.steering_goal
        if abs(steering_goal) > self.v.steering_max:
            if steering_goal > 0:
                steering_goal = self.v.steering_max
            else:
                steering_goal = -self.v.steering_max
        else:
          self.steering_goal = steering_goal

def Test_Helmsman_Node():
    h = helmsman()
    h.Connect()
    h.Loop()
    h.Disconnect()

if __name__ == '__main__':
    #Test_Mqtt_Node()
    Test_Helmsman_Node()
