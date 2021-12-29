#! /usr/bin/env python

import vtk
from vtkmodules.all import *
from vtkmodules.util import numpy_support
import sys, os, numpy


def ball_actor():
    ball = vtkSphereSource()
    ball.SetCenter(-0.8, -1, 6.3)
    ball.SetRadius(0.8)
    ball.SetThetaResolution(100)
    ball.SetPhiResolution(100)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(ball.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)

    prop = actor.GetProperty()
    prop.SetOpacity(0.1)

    return actor

def cylinder_actor():
    cylinder = vtkCylinderSource()
    cylinder.SetCenter(0.5, 0.5, 0.5)
    cylinder.SetHeight(1)
    cylinder.SetRadius(0.5)
    cylinder.SetResolution(100)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(cylinder.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)

    transform = vtkTransform()
    transform.PostMultiply()
    transform.RotateX(90.0)
    transform.Translate(0, 1, 0)

    actor.SetUserTransform(transform)

    prop = actor.GetProperty()
    prop.SetOpacity(0.2)

    return actor

def cube_actor(c1,c2):
    cube = vtkCubeSource()
    x1,y1,z1 = c1
    x2,y2,z2 = c2
    cube.SetBounds(x1,x2,y1,y2,z1,z2)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(cube.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)

    prop = actor.GetProperty()
    prop.SetOpacity(0.8)
    prop.SetColor(0.0, 1.0, 1.0)

    return actor


def text_actor(time):
    actor = vtk.vtkTextActor()
    actor.SetInput(time)
    actor.SetDisplayPosition(20, 30)

    prop = actor.GetTextProperty()
    prop.SetFontFamilyToCourier()
    prop.SetFontSize(15)
    prop.SetColor(1, 1, 1)

    global file_label
    file_label = actor

    return actor


def particles_actor(reader,camera,data_type,thres=False):
    aa = vtkAssignAttribute()
    if data_type == 'fpm':
        aa.Assign("concentration", vtkDataSetAttributes.SCALARS, vtkAssignAttribute.POINT_DATA)
    elif data_type == 'jet3b':
        aa.Assign("rho", vtkDataSetAttributes.SCALARS, vtkAssignAttribute.POINT_DATA)
    elif data_type == 'cos':
        aa.Assign("phi", vtkDataSetAttributes.SCALARS, vtkAssignAttribute.POINT_DATA)
    if isinstance(reader,vtkXMLDataReader):
        aa.SetInputConnection(reader.GetOutputPort())
    else:
        aa.SetInputData(reader)
    

    mask = vtkMaskPoints()
    mask.SetOnRatio(1)
    mask.SetInputConnection(aa.GetOutputPort())
    mask.GenerateVerticesOn()
    mask.SingleVertexPerCellOn()
    
    if data_type == 'fpm' and thres:
        threshold = vtkThreshold()
        threshold.SetInputConnection(mask.GetOutputPort())
        threshold.ThresholdByUpper(0.23)
        geom = vtkGeometryFilter()
        geom.SetInputConnection(threshold.GetOutputPort())
    else:
        geom = vtkGeometryFilter()
        geom.SetInputConnection(mask.GetOutputPort())

    sort = vtkDepthSortPolyData()
    sort.SetInputConnection(geom.GetOutputPort())
    sort.SetDirectionToBackToFront()
    sort.SetCamera(camera)

    lut = vtkLookupTable()
    lut.SetNumberOfColors(256)
    lut.SetHueRange( 4. / 6.,0)
    lut.Build()

    mapper = vtkDataSetMapper()
    mapper.SetInputConnection(sort.GetOutputPort())
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(0.1, 0.9)
    mapper.SetScalarModeToUsePointData()
    mapper.ScalarVisibilityOn()

    actor = vtkActor()
    actor.SetMapper(mapper)

    prop = actor.GetProperty()
    # prop.SetPointSize(1.5)
    # prop.SetOpacity(1)
    prop.SetRenderPointsAsSpheres(1)
    prop.SetPointSize(8)

    return actor

# def load_show(filename):
#     reader = vtkXMLUnstructuredGridReader()
#     reader.SetFileName(filename)
#     reader.Update()
#     time = reader.GetOutput().GetFieldData().GetArray("time").GetTuple1(0)
#     step = reader.GetOutput().GetFieldData().GetArray("step").GetTuple1(0)

#     camera = vtkCamera()
#     camera.SetPosition(0, -25.0, 12.5)
#     camera.SetFocalPoint(0, 0, 4.1)

#     renderer = vtkRenderer()
#     renderer.SetBackground(0, 0, 0)
#     renderer.SetActiveCamera(camera)


#     renderer.AddActor(particles_actor(reader,camera))
#     renderer.AddActor(cylinder_actor())
#     renderer.AddActor(text_actor(str(step)))

#     window = vtkRenderWindow()
#     window.PointSmoothingOn()
#     window.SetSize(512, 512)
#     # window.SetOffScreenRendering(1)
#     window.AddRenderer(renderer)

#     renWinInter = vtkRenderWindowInteractor()
#     renWinInter.SetRenderWindow(window)


#     print(str(filename) + " " + str(time) + " " + str(step))

#     file_label.SetInput("%s: %02.3fs (%03d)" % (os.path.dirname(filename), time, step))

#     window.Render()
#     renWinInter.Start()

def show(data,time="TEST",c1=None,c2=None,outfile="test",show=True,data_type='fpm',thres=False):
    camera = vtkCamera()
    camera.SetPosition(0, -2.5, 1.25)
    camera.SetFocalPoint(0.5, 0.5, 0.5)
    camera.SetViewUp(0,0,1)

    renderer = vtkRenderer()
    renderer.SetBackground(0.32, 0.34, 0.43)
    # renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.SetActiveCamera(camera)

    renderer.AddActor(particles_actor(data,camera,data_type,thres))
    if data_type == 'fpm':
        renderer.AddActor(cylinder_actor())
    if c1 is not None:
        renderer.AddActor(cube_actor(c1,c2))
    renderer.AddActor(text_actor(time))

    window = vtkRenderWindow()
    window.PointSmoothingOn()
    window.SetSize(512, 512)
    # window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)

    renWinInter = vtkRenderWindowInteractor()
    renWinInter.SetRenderWindow(window)

    window.Render()
    if show:
        renWinInter.Start()
    else:
        wtif = vtkWindowToImageFilter()
        wtif.SetInput(window)
        wtif.Update()

        pngwriter = vtkPNGWriter()
        pngwriter.SetInputConnection(wtif.GetOutputPort())
        pngwriter.SetFileName(outfile + '.png')
        pngwriter.Write()

