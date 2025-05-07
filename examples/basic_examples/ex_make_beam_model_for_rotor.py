"""Generate internal geometry"""
import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs
import ALBATROSS

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

    pusher_prop_blade_geometry = pusher_prop.create_subgeometry(search_names=["Rotor_9_blades, 0"])
    # pusher_prop_blade_geometry.plot(opacity=0.5)
    pusher_prop_blade = cd.aircraft.components.Blade(AR=1,S_ref=1,
                                                     geometry=pusher_prop_blade_geometry)
    top_index = 174
    bot_index = 175
    skin_indices = [top_index,bot_index]
    #can also get these indices by something like:
    # skin_indices = list(pusher_prop_blade.geometry.function_names.keys())

    #select surfaces to define as subgeometries
    top_geometry = pusher_prop_blade.create_subgeometry(search_names=[str(top_index)])
    bottom_geometry = pusher_prop_blade.create_subgeometry(search_names=[str(bot_index)])
    pusher_prop_skin_geometry = pusher_prop_blade.create_subgeometry(search_names=[str(idx) for idx in skin_indices])
    
    #create the front and rear spar surfaces
    #TODO: need to add indices to surface geometries names to be able to create subgeometries
    front_spar_geometry,rear_spar_geometry = pusher_prop_blade.create_internal_geometry(
        top_geometry,
        bottom_geometry,
        spar_locations=np.array([0.15,0.65])
    )
    spar_indices = [list(spar.function_names.keys())[0] for spar in [front_spar_geometry,rear_spar_geometry]]
    pusher_prop_spar_geometry = pusher_prop_blade.create_subgeometry(search_names=[str(idx) for idx in spar_indices])

    #view the blade with the new spar surfaces
    # pusher_prop_blade.geometry.plot(opacity=0.5)
    pusher_prop_blade_geometry.evaluate(([[174,(i/10,0)] for i in range(11)]+
                                          [[174,(i/10,1)] for i in range(11)]),plot=True)
    pusher_prop_blade_geometry.evaluate([[174,(i/10,0)] for i in range(11)])

    #MATERIALS
    #Aluminum
    E_Al = csdl.Variable(value=69E9, name='E_Al')
    G_Al = csdl.Variable(value=26E9, name='G_Al')
    density_Al = csdl.Variable(value=2700, name='density_Al')
    nu_Al = csdl.Variable(value=0.33, name='nu_Al')
    aluminum = cd.materials.IsotropicMaterial(name='Aluminum', density=density_Al, E=E_Al, nu=nu_Al, G=G_Al)

    #Polyurethane Foam 
    E_foam = csdl.Variable(value=7E8, name='E_foam')
    G_foam = csdl.Variable(value=6E8, name='G_foam')
    density_foam = csdl.Variable(value=400, name='density_foam')
    nu_foam = csdl.Variable(value=0.3, name='nu_foam')
    polyurethane_foam = cd.materials.IsotropicMaterial(name='polyurenthane_foam', density=density_foam, E=E_foam, nu=nu_foam, G=G_foam)

    #DEFINE SURFACE THICKNESSES
    #create a constant function space for thicknesses 
    fxn_space = lfs.ConstantSpace(2) #change if you want variable 
    
    #create functionspace for skin thickness
    skin_t_fxn_space = pusher_prop_skin_geometry.create_parallel_space(fxn_space)
    skin_thickness = 0.001
    skin_thickneses=skin_thickness*np.ones((len(skin_indices),1))
    skin_coeffs,skin_fxn_set = skin_t_fxn_space.initialize_function(1,skin_thickneses)

    #create functionspaces for spar for spar caps
    #TODO: change this to a conditional functionspace based on spar locations
    spar_cap_t_fxn_space = pusher_prop_skin_geometry.create_parallel_space(fxn_space)
    spar_cap_thickness = 0.005
    spar_cap_thickneses=spar_cap_thickness*np.ones((len(skin_indices),1))
    spar_cap_coeffs,spar_cap_fxn_set = spar_cap_t_fxn_space.initialize_function(1,spar_cap_thickneses)

    #create functionspaces for spar 
    spar_t_fxn_space = pusher_prop_spar_geometry.create_parallel_space(fxn_space)
    spar_thickness = 0.005
    spar_thickneses=spar_thickness*np.ones((len(spar_indices),1))
    spar_coeffs,spar_fxn_set = spar_t_fxn_space.initialize_function(1,spar_thickneses)

    #set materials and thickness for each surface of which a curve will be extracted to 
    #   construct the cross-section (e.g. skin, spar box)
    #   foam fills will be assigned later
    # pusher_prop_blade.quantities.material_properties.set_material(
    #     material=aluminum,
    #     thickness=skin_fxn_set)

    # pusher_prop_blade.quantities.material_properties.set_material(
    #     material=aluminum,
    #     thickness=spar_fxn_set)

    pusher_prop_blade.quantities.material_properties.add_material(
        material=aluminum,
        thickness=skin_fxn_set,
        surface_indices=skin_indices
    )

    pusher_prop_blade.quantities.material_properties.add_material(
        material=aluminum,
        thickness=spar_cap_fxn_set,
        surface_indices=skin_indices
    )
    
    pusher_prop_blade.quantities.material_properties.add_material(
        material=aluminum,
        thickness=spar_fxn_set,
        surface_indices=spar_indices
    )
    # #========== HOW TO EVALUATE SEPARATE THICKNESSES FOR THE SKIN AND SPAR CAP ========#
    # #we want to use the .evaluate_stack() command, but there is an issue with this function
    # #we can make a mod to this function, in the var_groups in utils --> struct_utils:
    # #A sketch (look at .evaluate_thickness for a good template)
    # #get the material stack for a given surface:
    material_stack = pusher_prop_blade.quantities.material_properties.get_material_stack(174)

    # #CRITICAL: convert parametric coordinate to np.array, will not work without doing this:
    # parametric_coords = np.array(parametric_coords, dtype='O,O')
    # #TODO: Q: how does this need to be returned? --> A: as a csdl variable (array)
    # material_thicknesses = [] #swap for 
    # for material in material_stack:
    #     material_thickness = material['thickness'].evaluate(parametric_coords)
    #     material_thicknesses.append(material_thickness)

    #construct the surfaces for each cross-section
    #TODO: break this function into a few separate parts
    pusher_prop_blade.create_beam_xs_meshes(top_geometry=top_geometry,
                                            bottom_geometry=bottom_geometry,
                                            front_spar_geometry=front_spar_geometry, 
                                            rear_spar_geometry=rear_spar_geometry,
                                            num_spanwise=10)
    
    #construct the meshes for each surface in each cross-section

    #project each mesh for each surface onto the fit cross-sectional surfaces



    exit()


    # ================== LIFT ROTORS =========================== #
    #make plot of upper and lower skin surfaces for illustrative purposes:
    for i in range(10):
        pusher_prop_blade_xs_skin = pusher_prop_blade.create_subgeometry(search_names=["skin_"+str(i)])
        
        xs_plot=pusher_prop_blade_xs_skin.plot(color=000000)

        # pusher_prop_blade_xs_ts.plot(additional_plotting_elements=[xs_bs_plot],opacity=0.5)

    pusher_prop_blade.geometry.plot(opacity=0.5)

    aluminum_albatross = ALBATROSS.material.caddee_material_to_albatross(aluminum)

    meshes = []

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
