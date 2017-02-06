import time

from pyfirmata import Arduino, util

import paho.mqtt.client as mqtt

import picamera

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
    """
    def __init__(self):
        self.board = Arduino('/dev/ttyUSB0')
        self.motor = self.board.get_pin('d:9:s')
        self.mot_stop = 90
        self.mot_slow_f = 100
        self.mot_slow_b = 80
        self.steering = self.board.get_pin('d:10:s')
        self.st_straight = 90
        self.speed_increment = 1	# a reasonable quantity for "go a bit faster"
        self.max_speed = 13411		# 30mph / 13.4112 meters/second
        self.camera = picamera.PiCamera()
        self.camera.vflip = True

class helmsman(mqtt_node):
    def __init__(self):
        super().__init__(Subscriptions=('set_speed', 'take_pic'), Blocking=False)
        self.v = vehicle()
        self.speed_goal = 0		# (int) mm/sec

    def rmsg_take_pic(self, msg):
        self.v.camera.capture('temp/test.jpg')

    def rmsg_set_speed(self, msg):
        self.GetGoalSpeed(msg)
        print(self.speed_goal)

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
        pass

    def GetGoalSpeed(self, speed_request):
        if speed_request in '+=':
          speed_goal = self.speed_goal + self.v.speed_increment
        elif speed_request == '-':
          speed_goal = self.speed_goal - self.v.speed_increment
        elif speed_request == 'z':
          speed_goal = 0
        else:
          try:
            speed_goal = int(speed_request)
          except:
            print("Bad Input '%s'" %(speed_request))
            speed_goal = self.speed_goal
        if speed_goal < 0:
            self.speed_goal = 0
        elif speed_goal > self.v.max_speed:
            self.speed_goal = self.v.max_speed
        else:
            self.speed_goal = speed_goal

def Test_Helmsman_Node():
    h = helmsman()
    h.Connect()
    h.Loop()
    h.Disconnect()

if __name__ == '__main__':
    #Test_Mqtt_Node()
    Test_Helmsman_Node()
