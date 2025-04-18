from __future__ import print_function
import socket

hostname = socket.gethostname()
print("Local hostname: %s" % hostname)

if hostname == "cube4":
    hostname = "cube4.ge.in.tum.de"
    print("Recognized as cube4")