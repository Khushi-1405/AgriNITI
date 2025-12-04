# turtle heart animation
import math
from turtle import *
def heart(i):
  return 15*math.sin(i)**3
def heartb(i):
  return 12*math.cos(i)-5*\
  math.cos(2*i)-2*\
  math.cos(3*i)-\
  math.cos(4*i)
speed(9000)
bgcolor("black")  

for i in range(6000):
  goto(heart(i)*20, heartb(i)*20)
  for j in range(5):
    color("red")
    goto(0,0)
done()    