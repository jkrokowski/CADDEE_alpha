"""Generate internal geometry"""
import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

recorder = csdl.Recorder(inline=True)
recorder.start()

caddee = cd.CADDEE()

make_meshes = False

# Import L+C .stp file and convert control points to meters
lpc_geom = cd.import_geometry("LPC_final_custom_blades.stp", scale=cd.Units.length.foot_to_m)

def mesh_rotor_blade(caddee : cd.CADDEE):
    """Build the base configuration."""
    
    # system component & airframe
    aircraft = cd.aircraft.components.Aircraft(geometry=lpc_geom)
    airframe = aircraft.comps["airframe"] = cd.Component()

    # ::::::::::::::::::::::::::: Make components :::::::::::::::::::::::::::
    # # Fuselage
    # fuselage_geometry = aircraft.create_subgeometry(search_names=["Fuselage"])
    # fuselage = cd.aircraft.components.Fuselage(length=10., geometry=fuselage_geometry)
    # airframe.comps["fuselage"] = fuselage

    # Main wing
    wing_geometry = aircraft.create_subgeometry(search_names=["Wing_1"])
    wing = cd.aircraft.components.Wing(AR=12.12, S_ref=19.6, taper_ratio=0.2, sweep=np.deg2rad(-20),
                                       geometry=wing_geometry, tight_fit_ffd=True)
    # wing.geometry.plot(opacity=0.3)
    # exit()
    top_surface_inds = [75, 79, 83, 87]
    top_geometry = wing.create_subgeometry(search_names=[str(i) for i in top_surface_inds])
    bottom_geometry = wing.create_subgeometry(search_names=[str(i+1) for i in top_surface_inds])
    wing.construct_ribs_and_spars(
        wing_geometry,
        top_geometry=top_geometry,
        bottom_geometry=bottom_geometry,
        num_ribs=9,
        LE_TE_interpolation="ellipse",
        plot_projections=False, 
        export_wing_box=False,
        full_length_ribs=True,
        spanwise_multiplicity=10,
        num_rib_pts=10,
        offset=np.array([0.,0.,.15]),
        finite_te=True,
        exclute_te=True
    )
    # wing.geometry.plot(opacity=0.3)
    # wing.geometry.export_iges("lpc_wing_geometry.igs")
    
    airframe.comps["wing"] = wing

    # base_config = cd.Configuration(system=aircraft)
    # # base_config.setup_geometry()
    # caddee.base_configuration = base_config
    # # wing.geometry.plot(opacity=0.3)

    # # exit()

    # # Horizontal tail
    # h_tail_geometry = aircraft.create_subgeometry(search_names=["Tail_1"])
    # h_tail = cd.aircraft.components.Wing(AR=4.3, S_ref=3.7, 
    #                                      taper_ratio=0.6, geometry=h_tail_geometry)
    # airframe.comps["h_tail"] = h_tail

    # Pusher prop
    pusher_prop_geometry = aircraft.create_subgeometry(search_names=[
        "Rotor-9-disk",
        "Rotor_9_blades",
        "Rotor_9_Hub",
    ])
    pusher_prop = cd.aircraft.components.Rotor(radius=2.74/2.5, geometry=pusher_prop_geometry)
    airframe.comps["pusher_prop"] = pusher_prop

    # pusher_prop.geometry.plot()

    pusher_prop_blade_geometry = pusher_prop.create_subgeometry(search_names=["Rotor_9_blades, 0"])
    # pusher_prop_blade_geometry.plot()
    pusher_prop_blade = cd.aircraft.components.Blade(AR=1,S_ref=1,
                                                     geometry=pusher_prop_blade_geometry)
    top_index = 174
    bot_index = 175

    top_geometry = pusher_prop_blade.create_subgeometry(search_names=[str(top_index)])

    bottom_geometry = pusher_prop_blade.create_subgeometry(search_names=[str(bot_index)])
    front_spar_geometry,rear_spar_geometry = pusher_prop_blade.construct_cross_section(
        top_geometry,
        bottom_geometry,
        spar_locations=np.array([0.25,0.6])
    )

    fxn_space = lfs.ConstantSpace(2) #change if you want variable 
    blade_t_fxn_space = pusher_prop_blade.geometry.create_parallel_space(fxn_space)
    # pusher_prop_blade.geometry.functions or .function_names
    #TODO: need to assign different surfaces and handle tip/root caps (2 surfs each)
    surf_indices = list(pusher_prop_blade.geometry.function_names.keys())
    surf_thickneses=0.0005*np.ones((len(surf_indices),1))
    coeffs,fxn_set = blade_t_fxn_space.initialize_function(1,surf_thickneses)

    #define material
    E = csdl.Variable(value=69E9, name='E')
    G = csdl.Variable(value=26E9, name='G')
    density = csdl.Variable(value=2700, name='density')
    nu = csdl.Variable(value=0.33, name='nu')
    aluminum = cd.materials.IsotropicMaterial(name='Aluminum', density=density, E=E, nu=nu, G=G)

    #set material for each surface
    pusher_prop_blade.quantities.material_properties.set_material(
        material=aluminum,
        thickness=fxn_set)
  
    pusher_prop_blade.create_beam_xs_meshes(top_geometry=top_geometry,
                                            bottom_geometry=bottom_geometry,
                                            front_spar_geometry=front_spar_geometry,
                                            rear_spar_geometry=rear_spar_geometry,
                                            num_spanwise=10)
    # pusher_prop_geometry.create_subset()
    # pusher_prop_geometry.function_names
    # pusher_prop_geometry.get_function_indices('Rotor_9_blades')

    # Lift rotors
    lift_rotors = []
    for i in range(8):
        rotor_geometry = aircraft.create_subgeometry(search_names=[
            f"Rotor_{i+1}_disk",
            f"Rotor_{i+1}_Hub",
            f"Rotor_{i+1}_blades",]
        )
        rotor = cd.aircraft.components.Rotor(radius=3.048/2.5, geometry=rotor_geometry)
        lift_rotors.append(rotor)
        airframe.comps[f"rotor_{i+1}"] = rotor
    lift_rotor_1 = airframe.comps['rotor_1']
    # lift_rotor_1.plot()
    
    lift_rotor_1_blade_geometry = lift_rotor_1.create_subgeometry(search_names="Rotor_1_blades, 0")
    lift_rotor_1_blade_geometry.plot()

    lift_rotor_1_blade = cd.aircraft.components.Blade(AR=1,S_ref=1,geometry=lift_rotor_1_blade_geometry)
    # lift_rotor_1_blade.create_beam_xs_meshes(,num_spanwise=15)

    # lift_rotor_1_blade = cd.aircraft.components.Rotor(radius=3.048/2.5,geometry=lift_rotor_1_blade_geometry)
    # lift_rotor_1_blade._ffd_block.plot()

    # u_coords = np.linspace(0,1,10)
    # v_coords = np.linspace(0,1,10)
    # index = 418
    # xcs = []
    # for u_coord in u_coords:
    #     parametric_coordinates = [(index, np.array([u_coord, v_coord])) for v_coord in v_coords]
    #     parametric_coordinates_2 = [(index+1, np.array([u_coord, v_coord])) for v_coord in v_coords]

    #     pts1 = lift_rotor_1_blade_geometry.evaluate(parametric_coordinates, plot=False)
    #     pts2 = lift_rotor_1_blade_geometry.evaluate(parametric_coordinates_2, plot=False)
    #     # pts = lift_rotor_1_blade_geometry.evaluate(parametric_coordinates+parametric_coordinates_2, plot=False)
        
    #     #remove duplicates but keep sorted
    #     # vals,indices= np.unique(pts.value,axis=0,return_index=True)
    #     # pts = np.concatenate([pts[sorted(indices),:].value,pts[0:1,:].value],axis=0)
        
    #     #close the loop
        

    #     xcs.append([pts1,pts2])

    # for xc_pts1,xc_pts2 in xcs:
    #     gmsh.initialize()
        
    #     lc =0.001
    #     pts1=[]
    #     pts2=[]
    #     for pt1,pt2 in zip(xc_pts1.value,xc_pts2.value):
    #         # pts.append(gmsh.model.geo.add_point(pt[0],pt[1],pt[2],lc))
    #         pts1.append(gmsh.model.occ.add_point(pt1[0],pt1[1],pt1[2]))
    #         pts2.append(gmsh.model.occ.add_point(pt2[0],pt2[1],pt2[2]))
    #     #check that xc_pts1 and 2 have the same start and end points
    #     # if, not add it to one of the lower surface
    #     if not np.equal(xc_pts1[0,:].value,xc_pts2[-1,:].value).all():
    #         pt = xc_pts1[0,:].value
    #         pts2.append(gmsh.model.occ.add_point(pt[0],pt[1],pt[2]))
    #     if not np.equal(xc_pts1[-1,:].value,xc_pts2[0,:].value).all():
    #         pt = xc_pts1[-1,:].value
    #         pts2.insert(0,gmsh.model.occ.add_point(pt[0],pt[1],pt[2]))

    #     # pts.append(pts[0])

    #     # spline = gmsh.model.geo.add_spline(pts)
    #     spline1 = gmsh.model.occ.add_spline(pts1)
    #     spline2 = gmsh.model.occ.add_spline(pts2)
    #     # bspline = gmsh.model.geo.add_compound_bspline([spline])

    #     # gmsh.model.geo.add_curve_loop([spline])
    #     CL1 = gmsh.model.occ.add_curve_loop([spline1,spline2])

    #     #try to offset, but if offset returns nothing, make a solid section
    #     CL2_tags_raw = gmsh.model.occ.offset_curve(CL1,-.005)

    #     # gmsh.model.geo.add_plane_surface([spline])
    #     # gmsh.model.occ.add_plane_surface([CL1])
    #     # gmsh.model.occ.add_plane_surface([CL2])

    #     #if the offset curve process fails, create a solid section
    #     try:
    #         #need to get proper list of curves by removing dim part of each tuple
    #         CL2_tags = [curve[1] for curve in CL2_tags_raw ]
    #         CL2 = gmsh.model.occ.add_curve_loop(CL2_tags)
    #         gmsh.model.occ.add_plane_surface([CL1,CL2])
    #     except:
    #         gmsh.model.occ.add_plane_surface([CL1])

    # gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)
    # gmsh.option.setNumber('Mesh.MeshSizeMax', 0.005)

    # # gmsh.model.geo.synchronize()
    # gmsh.model.occ.synchronize()

    # gmsh.model.mesh.generate(2)

    # gmsh.fltk.run()

    # gmsh.finalize()

    # CP1 = lift_rotor_1_blade_geometry.evaluate(lift_rotor_1_blade._corner_point_1).value
    # CP2 = lift_rotor_1_blade_geometry.evaluate(lift_rotor_1_blade._corner_point_2).value
    # CP3 = lift_rotor_1_blade_geometry.evaluate(lift_rotor_1_blade._corner_point_3).value
    # CP4 = lift_rotor_1_blade_geometry.evaluate(lift_rotor_1_blade._corner_point_4).value
    
    # #make a bounding box? then find primary axis, then do projections
    # ffd_block_corners_parametric = np.array([[0,0,0],
    #                               [0,0,1],
    #                               [0,1,0],
    #                               [0,1,1],
    #                               [1,0,0],
    #                               [1,0,1],
    #                               [1,1,0],
    #                               [1,1,1]],dtype=float)                         
                                  
    # ffd_block_corners=lift_rotor_1_blade._ffd_block.evaluate(parametric_coordinates=ffd_block_corners_parametric)
    # # print(ffd_block_corners.value)
    
    # #plot the ffd block and points for the selected lift rotor blade
    # lift_rotor_1_blade._ffd_block.plot()
    
    # #TODO: this works well if the beam axis is aligned with one of the primary directions, but is more challenging when it is not
    # #get primary beam axis 
    # idx = np.argmax(np.abs(np.min(ffd_block_corners.value,axis=0)-np.max(ffd_block_corners.value,axis=0)))

    # #get value of max and min along primary axis:
    # axis_max = np.max(ffd_block_corners.value[:,idx])
    # axis_min = np.min(ffd_block_corners.value[:,idx])

    # corner_set_1 = ffd_block_corners.value[ffd_block_corners.value[:,0] == axis_max,:][:,[i != 0 for i in range(3)]]
    # corner_set_2 = ffd_block_corners.value[ffd_block_corners.value[:,0] == axis_min,:][:,[i != 0 for i in range(3)]]

    # inplane_coords_1 = np.average(corner_set_1,axis=0)
    # inplane_coords_2 = np.average(corner_set_2,axis=0)

    # axis_start = np.hstack((axis_max,inplane_coords_1))
    # axis_end = np.hstack((axis_min,inplane_coords_2))

    # #beam axis points don't need to be projected to the rotor blade geometry
    # numspanwise = 10
    # beam_axis_pts = np.linspace(axis_start,axis_end,numspanwise+1)
    # #build a series of "bands" around the FFD block and project down onto the OML
    # numcircum = 8
    # #need to be sorted to make sure adjacent corners are the only ones that are being used
    # xs_pts_inplane = np.concatenate((np.linspace(corner_set_1[0,:],corner_set_1[1,:],numcircum),
    #                          np.linspace(corner_set_1[0,:],corner_set_1[2,:],numcircum),
    #                          np.linspace(corner_set_1[3,:],corner_set_1[1,:],numcircum),
    #                          np.linspace(corner_set_1[3,:],corner_set_1[2,:],numcircum)) )
    
    # xs_beam_axis_coord=np.tile(axis_start[idx],(xs_pts_inplane.shape[0],1))
    
    # xs_pts = np.concatenate((xs_beam_axis_coord,
    #                          xs_pts_inplane),axis=1)

    # # beam_axis_parametric = lift_rotor_1_blade_geometry.project(beam_axis_pts, plot=True, grid_search_density_parameter=10)
    # xs_parametric  = lift_rotor_1_blade_geometry.project(xs_pts, plot=True, grid_search_density_parameter=12)
    
    # # xs_parametric  = lift_rotor_1_blade_geometry.project(axis_start, direction=np.array([0,0,1]),plot=True, grid_search_density_parameter=10)

    # def generate_circle_3d(center, radius, num_points=100,axis=2):
    #     # Center of the circle
    #     x_center, y_center, z_center = center
        
    #     # Array of angles
    #     angles = np.linspace(0, 2 * np.pi, num_points)
    #     if axis==0:
    #         # Coordinates of the circle in 3D space
    #         z = z_center + radius * np.cos(angles)
    #         y = y_center + radius * np.sin(angles)
    #         x = np.full_like(z, x_center)  # X coordinate is constant (x_center)
    #     elif axis==1:
    #         # Coordinates of the circle in 3D space
    #         x = x_center + radius * np.cos(angles)
    #         z = z_center + radius * np.sin(angles)
    #         y = np.full_like(x, y_center)  # Z coordinate is constant (y_center)
    #     elif axis==2:
    #         # Coordinates of the circle in 3D space
    #         x = x_center + radius * np.cos(angles)
    #         y = y_center + radius * np.sin(angles)
    #         z = np.full_like(x, z_center)  # Z coordinate is constant (z_center)
        
    #     # Combine into an array of 3D points
    #     circle_points = np.vstack((x, y, z)).T
        
    #     return circle_points

    # radius = np.linalg.norm(np.max(corner_set_1,axis=0)-np.min(corner_set_1,axis =0))

    # xs_circle = generate_circle_3d(axis_start,radius,num_points=20,axis=idx)

    # for pt in xs_circle:
    #     direction = axis_start-pt
    #     pt_projection  = lift_rotor_1_blade_geometry.project(pt, direction=direction,plot=True, grid_search_density_parameter=10)

    
    exit()

    # Booms
    for i in range(8):
        boom_geometry = aircraft.create_subgeometry(search_names=[
            f"Rotor_{i+1}_Support",
        ])
        boom = cd.Component(geometry=boom_geometry)
        airframe.comps[f"boom_{i+1}"] = boom

    lpc_geom.plot(opacity=.5)

    # ::::::::::::::::::::::::::: Make meshes :::::::::::::::::::::::::::
    if make_meshes:
    # wing + tail
        # vlm_mesh = cd.mesh.VLMMesh()
        # wing_chord_surface = cd.mesh.make_vlm_surface(
        #     wing, 32, 15, LE_interp="ellipse", TE_interp="ellipse", ignore_camber=False,
        # )
        # vlm_mesh.discretizations["wing_chord_surface"] = wing_chord_surface
        
        # tail_surface = cd.mesh.make_vlm_surface(
        #     h_tail, 14, 6, ignore_camber=True
        # )
        # vlm_mesh.discretizations["tail_surface"] = tail_surface

        # lpc_geom.plot_meshes([wing_chord_surface.nodal_coordinates, tail_surface.nodal_coordinates])

        # rotors
        rotor_meshes = cd.mesh.RotorMeshes()
        # pusher prop
        pusher_prop_mesh = cd.mesh.make_rotor_mesh(
            pusher_prop, num_radial=30, num_azimuthal=1, num_blades=4, plot=True
        )
        rotor_meshes.discretizations["pusher_prop_mesh"] = pusher_prop_mesh

        # lift rotors
        for i in range(8):
            rotor_mesh = cd.mesh.make_rotor_mesh(
                lift_rotors[i], num_radial=30, num_blades=2,
            )
            rotor_meshes.discretizations[f"rotor_{i+1}_mesh"] = rotor_mesh
        
        # lpc_geom.plot_meshes([pusher_prop_mesh.nodal_coordinates])

    # aircraft.geometry.plot()

    # Make base configuration    
    base_config = cd.Configuration(system=aircraft)
    exit()
    base_config.setup_geometry()
    caddee.base_configuration = base_config

    # Store meshes
    if make_meshes:
        mesh_container = base_config.mesh_container
        # mesh_container["vlm_mesh"] = vlm_mesh
        mesh_container["rotor_meshes"] = rotor_meshes


mesh_rotor_blade(caddee)
