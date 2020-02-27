#! /usr/bin/env python

import vtk
from vtk import *
from vtk.util import numpy_support
import sys, os, numpy


if __name__ =="__main__":
    if len(sys.argv) < 2:
        print('creates one PNG image visualizing each of the vtu files passed as arguments')
        print('')
        print('usage:' + sys.argv[0] + "<file1>.vtu [<file2>.vtu ...]")
        print()
        sys.exit(0)


def cylinder_actor():
    cylinder = vtkCylinderSource()
    cylinder.SetCenter(0, 0, 0)
    cylinder.SetHeight(10);
    cylinder.SetRadius(5);
    cylinder.SetResolution(100);

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(cylinder.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)

    transform = vtkTransform()
    transform.PostMultiply()
    transform.RotateX(90.0)
    transform.Translate(0, 0, 5)

    actor.SetUserTransform(transform);

    prop = actor.GetProperty()
    prop.SetOpacity(0.2)

    return actor


def text_actor():
    actor = vtk.vtkTextActor()
    actor.SetInput('TEST')
    actor.SetDisplayPosition(20, 30)

    prop = actor.GetTextProperty()
    prop.SetFontFamilyToCourier()
    prop.SetFontSize(15)
    prop.SetColor(1, 1, 1)

    global file_label
    file_label = actor

    return actor


def particles_actor(reader,camera):
    aa = vtkAssignAttribute()
    aa.Assign("concentration", vtkDataSetAttributes.SCALARS, vtkAssignAttribute.POINT_DATA)
    aa.SetInputConnection(reader.GetOutputPort())

    mask = vtkMaskPoints()
    mask.SetOnRatio(1)
    mask.SetInputConnection(aa.GetOutputPort())
    mask.GenerateVerticesOn()
    mask.SingleVertexPerCellOn()

    threshold = vtkThreshold()
    threshold.SetInputConnection(mask.GetOutputPort())
    threshold.ThresholdByUpper(0)
    # threshold.ThresholdByLower(2)

    geom = vtkGeometryFilter()
    geom.SetInputConnection(threshold.GetOutputPort())

    sort = vtkDepthSortPolyData()
    sort.SetInputConnection(geom.GetOutputPort())
    sort.SetDirectionToBackToFront()
    sort.SetCamera(camera)

    lut = vtkLookupTable()
    lut.SetNumberOfColors(256)
    lut.SetHueRange(0, 4. / 6.)
    lut.Build()

    mapper = vtkDataSetMapper()
    mapper.SetInputConnection(sort.GetOutputPort())
    mapper.SetLookupTable(lut)
    # mapper.SetScalarRange(0, 2)
    mapper.SetScalarRange(0, 350)
    mapper.SetScalarModeToUsePointData()
    mapper.ScalarVisibilityOn()

    actor = vtkActor()
    actor.SetMapper(mapper)

    prop = actor.GetProperty()
    prop.SetPointSize(1.5)
    prop.SetOpacity(0.7)

    return actor

def show(filename):
    camera = vtkCamera()
    camera.SetPosition(0, -25.0, 12.5)
    camera.SetFocalPoint(0, 0, 4.1)

    renderer = vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    renderer.SetActiveCamera(camera)

    reader = vtkXMLUnstructuredGridReader()

    renderer.AddActor(particles_actor(reader,camera))
    renderer.AddActor(cylinder_actor())
    renderer.AddActor(text_actor())

    window = vtkRenderWindow()
    window.PointSmoothingOn()
    window.SetSize(512, 512)
    # window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)

    renWinInter = vtkRenderWindowInteractor()
    renWinInter.SetRenderWindow(window)

    outfile = os.path.splitext(filename)[0]

    reader.SetFileName(filename)
    reader.Update()

    time = reader.GetOutput().GetFieldData().GetArray("time").GetTuple1(0)
    step = reader.GetOutput().GetFieldData().GetArray("step").GetTuple1(0)

    print(str(filename) + " " + str(time) + " " + str(step))

    file_label.SetInput("%s: %02.3fs (%03d)" % (os.path.dirname(filename), time, step))

    window.Render()
    renWinInter.Start()

