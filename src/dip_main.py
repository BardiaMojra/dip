''' dip
  @author Bardia Mojra - 1000766739
  @brief ee-5323 - project -
  @date 10/31/21

  code based on below YouTube tutorial and Pymotw.com documentation for socket mod.
  @link https://www.youtube.com/watch?v=3QiPPX-KeSc
  @link https://pymotw.com/2/socket/tcp.html

  python socket module documentation
  @link https://docs.python.org/3/library/socket.html
  @link https://docs.python.org/3/howto/sockets.html

'''
import csv
import math
import numpy as np
import os
import pygame
import pyglet
from pyglet.window import key
import pymunk
import pymunk.constraints
import pymunk.pygame_util
import pandas as pd
import pyglet.gl as gl

''' custom libs
'''
import dvm
import tcm

''' NBUG
'''
from nbug import *


''' TEST CONFIG
'''
TEST_ID = 'Test 903'
SIM_DUR = 30.0 # in seconds
OUT_DIR = '../out/'
OUT_DATA = OUT_DIR+TEST_ID+'_data.csv'
CONF_DIR = '../config/'
# cart
m_c = 0.5
all_friction = 0.2
'''   pendulum 1   '''
l_1 = 0.4 # 6, 5, 4, 7 -- 4 ->
m_1 = 0.2 # 2, 3, 4 -- 1 -> stable
m_1_moment = 0.01
m_1_radius = 0.05
'''   pendulum 2  '''
l_2 = 0.7 # 6, 5, 7 -- 3 -> unstable
m_2 = 0.3 # 2, 3, 4 -- 2 -> unstable
m_2_moment = 0.001
m_2_radius = 0.05
# other config
output_labels=['t', 'x', 'dx', 'th_1', 'dth_1', 'th_2', 'dth_2']
# control config
# K gain matrix and Nbar found from modelling via Jupyter
# K = [16.91887353, 21.12423935, 137.96378003, -3.20040325, -259.72220049,  -50.48383455]
# Nbar = 17.0
K = [51.43763708,
     54.12690472,
     157.5467596,
     -21.67111679,
     -429.11603909,
     -88.73125241]
Nbar = 51.5

tConfig = tcm.test_configuration(TEST_ID=TEST_ID,
                                 OUT_DIR=OUT_DIR,
                                 OUT_DATA=OUT_DATA,
                                 CONF_DIR=CONF_DIR,
                                 SIM_DUR=SIM_DUR,
                                 output_labels=output_labels,
                                 all_friction=all_friction,
                                 cart_mass=m_c,
                                 pend_1_length=l_1,
                                 pend_1_mass=m_1,
                                 pend_1_moment=m_1_moment,
                                 pend_2_length=l_2,
                                 pend_2_mass=m_2,
                                 pend_2_moment=m_2_moment,
                                 K=K,
                                 Nbar=Nbar)

# log test config
tcm.pkl(tConfig)


''' MOD CONFIG
'''
SCREEN_WIDTH  = 700
SCREEN_HEIGHT = 500
# sim config
MAX_FORCE = 25
DT = 1 / 60.0
PPM = 200.0 # pxls per meter
END_ = 1000 # samples used for plotting and analysis
SHOW_ = True
cart_size = 0.3, 0.2

white_color = (0,0,0,0)
black_color = (255,255,255,255)
green_color = (0,135,0,255)
red_color   = (135,0,0,255)
blue_color  = (0,0,135,255)

''' main
'''
pygame.init()
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
# clock = pygame.time.Clock()
window = pyglet.window.Window(SCREEN_WIDTH, SCREEN_HEIGHT, vsync=False, caption='Double Inverted Pendulum Simulation')
gl.glClearColor(255,255,255,255)
# setup the space
space = pymunk.Space()
# options = pymunk.pygame_util.DrawOptions(surface)
# space.debug_draw(options)
space.gravity = 0, -9.81
# space.debug_draw(options)

fil = pymunk.ShapeFilter(group=1)

# screen.fill(pygame.Color("white"))
# options = pymunk.pygame_util.DrawOptions(screen)
# space.debug_draw(options)
# ground
ground = pymunk.Segment(space.static_body, (-4, -0.1), (4, -0.1), 0.1)
# ground.color = pygame.Color("pink")
ground.friction = all_friction
ground.filter = fil
space.add(ground)
# space.debug_draw(options)
# cart
cart_moment = pymunk.moment_for_box(m_c, cart_size)
cart_body = pymunk.Body(mass=m_c, moment=cart_moment)
cart_body.position = 0.0, cart_size[1] / 2
cart_shape = pymunk.Poly.create_box(cart_body, cart_size)
cart_shape.color = black_color
# cart_shape.color = red_color
# cart_shape.fill_color = red_color
# cart_shape.color = black_color
cart_shape.friction = ground.friction
space.add(cart_body, cart_shape)
# space.debug_draw(options)
# pendulum 1
pend_1_body = pymunk.Body(mass=m_1, moment=m_1_moment)
pend_1_body.position = cart_body.position[0], cart_body.position[1] + cart_size[1] / 2 + l_1
pend_shape = pymunk.Circle(pend_1_body, m_1_radius)
pend_shape.filter = fil
space.add(pend_1_body, pend_shape)

# joint
joint = pymunk.constraints.PivotJoint(cart_body, pend_1_body, cart_body.position + (0, cart_size[1] / 2))
joint.collide_bodies = False
space.add(joint)

# pendulum 2
pend_2_body = pymunk.Body(mass=m_2, moment=m_2_moment)
pend_2_body.position = cart_body.position[0], cart_body.position[1] + cart_size[1] / 2 + (2 * l_2)
pend_shape2 = pymunk.Circle(pend_2_body, m_2_radius)
pend_shape2.filter = fil
space.add(pend_2_body, pend_shape2)

# joint 2
joint2 = pymunk.constraints.PivotJoint(pend_1_body, pend_2_body, cart_body.position + (0, cart_size[1] / 2 + l_2))
joint2.collide_bodies = False
space.add(joint2)
# space.debug_draw(options)
print(f"cart mass = {cart_body.mass:0.1f} kg")
print(f"pendulum 1 mass = {pend_1_body.mass:0.1f} kg, pendulum moment = {pend_1_body.moment:0.3f} kg*m^2")
print(f"pendulum 2 mass = {pend_2_body.mass:0.1f} kg, pendulum moment = {pend_2_body.moment:0.3f} kg*m^2")



force = 0.0
ref = 0.0

color = (200, 200, 200, 200)
label_x = pyglet.text.Label(text='', font_size=12, color=color, x=10, y=SCREEN_HEIGHT - 28)
label_th_1 = pyglet.text.Label(text='', font_size=12, color=color, x=10, y=SCREEN_HEIGHT - 58)
label_th_2 = pyglet.text.Label(text='', font_size=12, color=color, x=10, y=SCREEN_HEIGHT - 88)
label_force = pyglet.text.Label(text='', font_size=12, color=color, x=10, y=SCREEN_HEIGHT - 118)

labels = [label_x, label_th_1, label_th_2, label_force]
# data recorder so we can compare our results to our predictions
if os.path.exists(OUT_DATA):
  os.remove(OUT_DATA)
with open(OUT_DATA, 'w') as f:
  output_header = str()
  for i, s in enumerate(output_labels):
    if i == 0:
      output_header = s
    else:
      output_header += ', '+s
  output_header += '\n'
  f.write(output_header)
  f.close()
currtime = 0.0
record_data = True

def draw_body(offset, body):
  for shape in body.shapes:
    if isinstance(shape, pymunk.Circle):
      vertices = []
      num_points = 10
      for ii in range(num_points):
        angle = ii / num_points * 2 * math.pi
        vertices.append(body.position + (shape.radius * math.cos(angle), shape.radius * math.sin(angle)))
      points = []
      for v in vertices:
        points.append(int(v[0] * PPM) + offset[0])
        points.append(int(v[1] * PPM) + offset[1])

      data = ('v2i', tuple(points))
      gl.glColor3b(255,255,255)
      pyglet.graphics.draw(len(vertices), pyglet.gl.GL_LINE_LOOP, data)
    elif isinstance(shape, pymunk.Poly):
      # get vertices in world coordinates
      vertices = [v.rotated(body.angle) + body.position for v in shape.get_vertices()]
      # convert vertices to pixel coordinates
      points = []
      for v in vertices:
        points.append(int(v[0] * PPM) + offset[0])
        points.append(int(v[1] * PPM) + offset[1])
      data = ('v2i', tuple(points))
      gl.glColor3b(255,255,255)
      pyglet.graphics.draw(len(vertices), pyglet.gl.GL_LINE_LOOP, data)

def draw_line_between(offset, pos1, pos2):
  vertices = [pos1, pos2]
  points = []
  for v in vertices:
    points.append(int(v[0] * PPM) + offset[0])
    points.append(int(v[1] * PPM) + offset[1])
  data = ('v2i', tuple(points))
  gl.glColor3b(255,255,255)
  pyglet.graphics.draw(len(vertices), pyglet.gl.GL_LINE_STRIP, data)

def draw_ground(offset):
  vertices = [v + (0, ground.radius) for v in (ground.a, ground.b)]
  # convert vertices to pixel coordinates
  points = []
  for v in vertices:
    points.append(int(v[0] * PPM) + offset[0])
    points.append(int(v[1] * PPM) + offset[1])
  data = ('v2i', tuple(points))
  pyglet.graphics.draw(len(vertices), pyglet.gl.GL_LINES, data)

@window.event
def on_draw():
  window.clear()
  # center view x around 0
  offset = (250, 5)
  draw_body(offset, cart_body)
  draw_body(offset, pend_1_body)
  draw_line_between(offset, cart_body.position + (0, cart_size[1] / 2), pend_1_body.position)
  draw_body(offset, pend_2_body)
  draw_line_between(offset, pend_1_body.position, pend_2_body.position)
  draw_ground(offset)
  for label in labels:
    label.draw()

@window.event
def on_key_press(symbol, modifiers):
    # Symbolic names:
    if symbol == key.ESCAPE:
      window.close()

def simulate(_):
  global currtime
  if currtime > SIM_DUR:
      window.close()

  # nprint('_',_)
  # ensure we get a consistent simulation step - ignore the input dt value
  dt = DT
  # simulate the world
  # NOTE: using substeps will mess up gains
  space.step(dt)
  # populate the current state
  posx = cart_body.position[0]
  velx = cart_body.velocity[0]
  th_1 = pend_1_body.angle
  th_1v = pend_1_body.angular_velocity
  th_2 = pend_2_body.angle
  th_2v = pend_2_body.angular_velocity
  # dump our data so we can plot
  if record_data:
    with open(OUT_DATA, 'a+') as f:
      f.write(f"{currtime:0.5f}, {posx:0.5f}, {velx:0.5f}, {th_1:0.5f}, {th_1v:0.5f}, {th_2:0.5f}, {th_2v:0.5f} \n")
      f.close()
    currtime += dt
  # calculate our gain based on the current state
  gain = K[0] * posx + K[1] * velx + K[2] * th_1 + K[3] * th_1v + K[4] * th_2 + K[5] * th_2v
  # calculate the force required
  global force
  force = ref * Nbar - gain
  # kill our motors if our angles get out of control
  if math.fabs(pend_1_body.angle) > 1.0 or math.fabs(pend_2_body.angle) > 1.0:
    force = 0.0
  # cap our maximum force so it doesn't go crazy
  if math.fabs(force) > MAX_FORCE:
    force = math.copysign(MAX_FORCE, force)
  # apply force to cart center of mass
  cart_body.apply_force_at_local_point((force, 0.0), (0, 0))

def update_state_label(_):
  '''
    function to store the current state to draw on screen
  '''
  label_x.text = f'x: {cart_body.position[0]:0.3f} m'
  label_th_1.text = f'theta_1: {pend_1_body.angle:0.3f} rad'
  label_th_2.text = f'theta_2: {pend_2_body.angle:0.3f} rad'
  label_force.text = f'force: {force:0.1f} N'

def update_reference(_, newref):
  global ref
  ref = newref

# callback for simulation
pyglet.clock.schedule_interval(simulate, DT)
pyglet.clock.schedule_interval(update_state_label, 0.25)

# schedule some small movements by updating our reference
pyglet.clock.schedule_once(update_reference, 2, 0.2)
pyglet.clock.schedule_once(update_reference, 7, 0.6)
pyglet.clock.schedule_once(update_reference, 12, 0.2)
pyglet.clock.schedule_once(update_reference, 17, 0.0)

pyglet.app.run()
f.close()



# data recorder so we can compare our results to our predictions
# f = open(OUT_DATA, 'r')
# ['t', 'x', 'dx', 'th_1', 'dth_1', 'th_2', 'dth_2', 'L1', 'L2']
# for i in test_IDs:
tConfig = tcm.unpkl(TEST_ID, CONF_DIR)
df = pd.read_csv(tConfig.out_data)
df = dvm.get_losses(df,
                    dataPath=tConfig.data_path,
                    lossPath=tConfig.loss_path)

# plot pose
# ['t', 'x', 'dx', 'th_1', 'dth_1', 'th_2', 'dth_2', 'L1', 'L2']
cols = [0, 1, 3, 5]
xy_df = df.iloc[:,cols].copy()
dvm.plot_df(xy_df,
            plot_title='State Position',
            labels=xy_df.columns,
            test_id=tConfig.id,
            out_dir=tConfig.out_dir,
            end=END_,
            show=SHOW_)

# plot vel
# ['t', 'x', 'dx', 'th_1', 'dth_1', 'th_2', 'dth_2', 'L1', 'L2']
cols = [0, 2, 4, 6]
xy_df = df.iloc[:,cols].copy()
dvm.plot_df(xy_df,
            plot_title='State Velocity',
            labels=xy_df.columns,
            test_id=tConfig.id,
            out_dir=tConfig.out_dir,
            end=END_,
            show=SHOW_)

# plot losses
# ['t', 'x', 'dx', 'th_1', 'dth_1', 'th_2', 'dth_2', 'L1', 'L2']
cols = [0, 7, 8]
xy_df = df.iloc[:,cols].copy()
dvm.plot_df(xy_df,
            plot_title='State Losses',
            labels=xy_df.columns,
            test_id=tConfig.id,
            out_dir=tConfig.out_dir,
            end=END_,
            show=SHOW_)

# print losses
dvm.print_losses(df)
