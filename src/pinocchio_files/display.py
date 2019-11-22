
# Example of a class Display that connect to Gepetto-viewer and implement a
# 'place' method to set the position/rotation of a 3D visual object in a scene.
# The gepetto-gui must already be running


# Typical header of a Python script using Pinocchio
import pinocchio as se3
from pinocchio.utils import *
import gepetto.corbaserver


class Display():
    '''
    Class Display: Example of a class implementing a client for the Gepetto-viewer server. The main
    method of the class is 'place', that sets the position/rotation of a 3D visual object in a scene.
    '''

    def __init__(self, windowName="pinocchio"):
        '''
        This function connect with the Gepetto-viewer server and open a window with the given name.
        If the window already exists, it is kept in the current state. Otherwise, the newly-created
        window is set up with a scene named 'world'.
        '''

        # Create the client and connect it with the display server.
        try:
            self.viewer = gepetto.corbaserver.Client()
        except:
            print( "Error while starting the viewer client. ")
            print( "Check whether Gepetto-viewer is properly started")

        # Open a window for displaying your model.
        try:
            # If the window already exists, do not do anything.
            windowID = self.viewer.gui.getWindowID(windowName)
            print( "Warning: window '" + windowName + "' already created.")
            print( "The previously created objects will not be destroyed and do not have to be created again.")
        except:
            # Otherwise, create the empty window.
            windowID = self.viewer.gui.createWindow(windowName)
            # Start a new "scene" in this window, named "world", with just a floor.
            self.viewer.gui.createSceneWithFloor("world")
            self.viewer.gui.addSceneToWindow("world", windowID)

        # Finally, refresh the layout to obtain your first rendering.
        self.viewer.gui.refresh()

    def nofloor(self):
        '''
        This function will hide the floor.
        '''
        self.viewer.gui.setVisibility('world/floor', "OFF")
        self.viewer.gui.refresh()

    def place(self, objName, M, refresh=True):
        '''
        This function places (ie changes both translation and rotation) of the object
        names "objName" in place given by the SE3 object "M". By default, immediately refresh
        the layout. If multiple objects have to be placed at the same time, do the refresh
        only at the end of the list.
        '''
        self.viewer.gui.applyConfiguration(objName,
                                           se3ToXYZQUAT(M))
        if refresh:
            self.viewer.gui.refresh()




def load_simple_objects():
    # Instantiate a Display object
    display = Display()

    # Example of use of the class Display to create a box visual object.
    # Attributes of the box
    boxid = 147
    name = 'box' + str(boxid)
    [w, h, d] = [1.0, 1.0, 1.0]
    color = [red, green, blue, transparency] = [1, 1, 0.78, 1.0]
    display.viewer.gui.addBox('world/' + name, w, h, d, color)

    # A sphere of dimension 1x1x1 can be created using:
    display.viewer.gui.addSphere('world/sphere', 1.0, color)

    # A cylinder of dimension 1x1x1 can be created using:
    # Define the attributes of the Cylinder
    radius = 1.0
    height = 1.0
    display.viewer.gui.addCylinder('world/cylinder', radius, height, color)


    # Other 3D primitives can be created through Gepetto-viewer.
    # For more information checkout idl in the terminal, ~$ less /opt/openrobots/share/idl/gepetto/corbaserver/graphical-interface.idl

    ########################################################
    ########################################################

    # Moving Objects
    """
    Place an object by simply mentioning its name and the placement you want.
    Gepetto-Viewer expects a XYZ-roll-pitch-yaw representation of the SE3 placement.
    The translation is performed by the method "place" of your class "Display".
    After placing an object, the layout must be explicitly refreshed.
    This is to reduce the display load when flushing several objects at the same time.
    Typically, during robot movements, you would only ask one refresh per control cycle.
    The method "Display.place" refreshes the layout by default.
    """

    display.place("world/box147", se3.SE3.Random(), False)
    display.place("world/sphere", se3.SE3.Random(), False)
    display.place("world/cylinder", se3.SE3.Random())


load_simple_objects()
