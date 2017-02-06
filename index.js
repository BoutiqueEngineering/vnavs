var express = require('express');
var fs = require('fs');
var path = require('path');
var spawn = require('child_process').spawn;
var proc;

var image_subdir = '/temp'
var image_path = image_subdir + '/test.jpg'
var image_fqn = '/home/pi/projects/vnavs' + image_path
var static_subdir = '/node_root'

// connect, message and disconnect are built-in socket.io event names
var socket_event_connect = 'connect'
var socket_event_disconnect = 'disconnect'
var socket_event_imageReady = 'imageReady'
var socket_event_startStream = 'startStream'

var app = express();
var http = require('http').Server(app);
var io = require('socket.io')(http);
app.use('/', express.static(path.join(__dirname, static_subdir)));
app.use(image_subdir, express.static(path.join(__dirname, image_subdir)));

app.get('/', function(req, res) {
  res.sendFile(__dirname + static_subdir + '/index.html');
});

var sockets = {};

io.on('connection', function(socket) {
  sockets[socket.id] = socket;
  console.log("Total clients connected : ", Object.keys(sockets).length);
  socket.on(socket_event_disconnect, function() {
    delete sockets[socket.id];
    // no more sockets, kill the stream
    if (Object.keys(sockets).length == 0) {
      app.set('watchingFile', false);
      if (proc) proc.kill();
      fs.unwatchFile(image_fqn);
    }
  });
  socket.on(socket_event_startStream, function() {
    startStreaming(io);
  });
});

http.listen(3000, function() {
  console.log('listening on *:3000');
});

function stopStreaming() {
  if (Object.keys(sockets).length == 0) {
    app.set('watchingFile', false);
    if (proc) proc.kill();
    fs.unwatchFile(image_fqn);
  }
}

function startStreaming(io) {
  if (app.get('watchingFile')) {
    io.sockets.emit(socket_event_imageReady, image_path + '?_t=' + (Math.random() * 100000));
    return;
  }

  //var args = ["-vf", "-w", "640", "-h", "480", "-o", "./stream/image_stream.jpg", "-t", "999999999", "-tl", "100"];
  //proc = spawn('raspistill', args);
  console.log('Watching for changes...');
  app.set('watchingFile', true);
  fs.watchFile(image_fqn, function(current, previous) {
    io.sockets.emit(socket_event_imageReady, image_path + '?_t=' + (Math.random() * 100000));
  })
}
