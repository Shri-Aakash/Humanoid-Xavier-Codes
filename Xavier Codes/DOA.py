from tuning import Tuning
import usb.core
import usb.util
import time
import socket
IP="10.4.1.45"
PORT=40000
ADDR=(IP,PORT)
 
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
def Connection(ang):
    print("Server Starting")
    client=socket.socket(family=socket.AF_INET,type=socket.SOCK_STREAM)
    client.connect(ADDR)
    #file=open('data/Message.txt','r')
    data=str(ang)
    
    client.send('Message.txt'.encode('utf-8'))
    msg=client.recv(1024).decode('utf-8')
    print(f'Server: {msg}')
    
    client.send(data.encode('utf-8'))
    msg=client.recv(1024).decode('utf-8')
    print(f'Server: {msg}')
    
    #file.close()
    client.close()

angle=123.456
if dev:
    Mic_tuning = Tuning(dev)
    print(Mic_tuning.direction)
    while True:
        try:
            print(Mic_tuning.direction)
            if Mic_tuning.direction==angle:
                continue
            else:
                Connection(Mic_tuning.direction)
                angle=Mic_tuning.direction
            time.sleep(1)
        except KeyboardInterrupt:
            break
