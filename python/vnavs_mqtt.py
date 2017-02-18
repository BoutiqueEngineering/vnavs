from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)
import sys
import os
import time
import ConfigParser

import paho.mqtt.client as mqtt

config_file_path = os.path.expanduser("~/vnavs.ini")
handler_method_prefix = 'rmsg_'

class mqtt_node(object):
    def __init__(self, Subscriptions=[], Blocking=False):
        self.config = ConfigParser.SafeConfigParser()
        self.config.readfp(open(config_file_path))
        self.blocking_mode = Blocking
        self.subscriptions = Subscriptions
        self.handlers = {}
        self.broker_host = self.config.get("MqttBroker", "Host")
        self.broker_port = int(self.config.get("MqttBroker", "Port"))	# 1883
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
            handler_name = handler_method_prefix + this_topic.replace('/', '_')
            handler_method = getattr(self, handler_name, None)
            if handler_method is None:
                print("No message handler for topic '%s'" % (this_topic))
            self.handlers[this_topic] = handler_method
            self.mqttc.subscribe(this_topic, 0)
        print(self.subscriptions)

    def on_message(self, client, userdata, message):
        print(message.topic + " " + str(message.qos) + " " + str(message.payload))
        handler_method = self.handlers[message.topic]
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

