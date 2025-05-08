import lsdo_function_spaces as lfs
from lsdo_function_spaces import FunctionSet
import CADDEE_alpha as cd
from CADDEE_alpha.core.component import Component
from CADDEE_alpha.core.aircraft.components.wing import Wing, WingParameters
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization, VolumeSectionalParameterizationInputs
)
import time
import numpy as np
from dataclasses import dataclass
import csdl_alpha as csdl
from typing import Union
import lsdo_geo as lg
import gmsh
import os
import matplotlib.pyplot as plt

path = os.getcwd()



class Blade(Wing):
    """The Blade component class.
    
    Parameters
    ----------
    - AR : aspect ratio
    - S_ref : reference area
    - span (None default)
    - dihedral (deg) (None default)
    - sweep (deg) (None default)
    - taper_ratio (None default)

    Note that parameters may be design variables for optimizaiton.
    If a geometry is provided, the geometry parameterization sovler
    will manipulate the geometry through free-form deformation such 
    that the wing geometry satisfies these parameters.

    Attributes
    ----------
    - parameters : data class storing the above parameters
    - geometry : b-spline set or subset containing the wing geometry
    - comps : dictionary for children components
    - quantities : dictionary for storing (solver) data (e.g., field data)
    """
    def __init__(
        self, 
        AR : Union[int, float, csdl.Variable, None], 
        S_ref : Union[int, float, csdl.Variable, None],
        span : Union[int, float, csdl.Variable, None] = None, 
        dihedral : Union[int, float, csdl.Variable, None] = None, 
        sweep : Union[int, float, csdl.Variable, None] = None, 
        taper_ratio : Union[int, float, csdl.Variable, None] = None,
        incidence : Union[int, float, csdl.Variable] = 0, 
        root_twist_delta : Union[int, float, csdl.Variable] = 0,
        tip_twist_delta : Union[int, float, csdl.Variable] = 0,
        geometry : Union[lfs.FunctionSet, None]=None,
        tight_fit_ffd: bool = False,
        orientation: str = "horizontal",
        **kwargs
    ) -> None:
        kwargs["do_not_remake_ffd_block"] = True
        super(Wing, self).__init__(geometry=geometry, **kwargs)
        
        # Do type checking 
        csdl.check_parameter(AR, "AR", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(S_ref, "S_ref", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(span, "span", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(dihedral, "dihedral", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(sweep, "sweep", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(incidence, "incidence", types=(int, float, csdl.Variable))
        csdl.check_parameter(taper_ratio, "taper_ratio", types=(int, float, csdl.Variable), allow_none=True)
        csdl.check_parameter(root_twist_delta, "root_twist_delta", types=(int, float, csdl.Variable))
        csdl.check_parameter(tip_twist_delta, "tip_twist_delta", types=(int, float, csdl.Variable))

        # Check if wing is over-parameterized
        if all(arg is not None for arg in [AR, S_ref, span]):
            raise Exception("Wing comp over-parameterized: Cannot specifiy AR, S_ref, and span at the same time.")
        # Check if wing is under-parameterized
        if sum(1 for arg in [AR, S_ref, span] if arg is None) >= 2:
            raise Exception("Wing comp under-parameterized: Must specify two out of three: AR, S_ref, and span.")
        
        if incidence is not None:
            if incidence != 0.:
                raise NotImplementedError("incidence has not yet been implemented")

        self._name = f"wing_{self._instance_count}"
        self._tight_fit_ffd = tight_fit_ffd
        self._orientation = orientation
        
        # Assign parameters
        self.parameters : WingParameters =  WingParameters(
            AR=AR,
            S_ref=S_ref,
            span=span,
            sweep=sweep,
            incidence=incidence,
            dihedral=dihedral,
            taper_ratio=taper_ratio,
            root_twist_delta=root_twist_delta,
            tip_twist_delta=tip_twist_delta,
        )

        # Compute MAC (i.e., characteristic length)
        if taper_ratio is None:
            taper_ratio = 1
        if AR is not None and S_ref is not None:
            lam = taper_ratio
            span = (AR * S_ref)**0.5
            root_chord = 2 * S_ref/((1 + lam) * span)
            MAC = (2/3) * (1 + lam + lam**2) / (1 + lam) * root_chord
            self.quantities.drag_parameters.characteristic_length = MAC
            self.parameters.MAC = MAC
        elif S_ref is not None and span is not None:
            lam = taper_ratio
            span = self.parameters.span
            root_chord = 2 * S_ref/((1 + lam) * span)
            MAC = (2/3) * (1 + lam + lam**2) / (1 + lam) * root_chord
            self.quantities.drag_parameters.characteristic_length = MAC
            self.parameters.MAC = MAC
        elif span is not None and AR is not None:
            lam = taper_ratio
            S_ref = span**2 / AR
            self.parameters.S_ref = S_ref
            root_chord = 2 * S_ref/((1 + lam) * span)
            MAC = (2/3) * (1 + lam + lam**2) / (1 + lam) * root_chord
            self.quantities.drag_parameters.characteristic_length = MAC
            self.parameters.MAC = MAC

        # Compute form factor according to Raymer 
        # (ignoring Mach number; include in drag build up model)
        x_c_m = self.parameters.thickness_to_chord_loc
        t_o_c = self.parameters.thickness_to_chord

        if t_o_c is None:
            t_o_c = 0.15
        if sweep is None:
            sweep = 0.

        FF = (1 + 0.6 / x_c_m + 100 * (t_o_c) ** 4) * csdl.cos(sweep) ** 0.28
        self.quantities.drag_parameters.form_factor = FF

        if self.geometry is not None:
            # Check for appropriate geometry type
            if not isinstance(self.geometry, (lfs.FunctionSet)):
                raise TypeError(f"wing gometry must be of type {lfs.FunctionSet}")
            else:
                t1 = time.time()
                # Set the wetted area
                self.parameters.S_wet = self.quantities.surface_area
                t2 = time.time()
                # print("time for computing wetted area", t2-t1)

                t3 = time.time()
                # Make the FFD block upon instantiation
                ffd_block = self._make_ffd_block(self.geometry, tight_fit=tight_fit_ffd, degree=(1, 2, 1), num_coefficients=(2, 2, 2))
                # ffd_block.plot()
                t4 = time.time()
                print("time for making ffd_block", t4-t3)
                # ffd_block.plot()

                t5 = time.time()
                # Compute the corner points of the wing 
                self._LE_base_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([1., 0., 1.])), plot=False, extrema=True)
                self._LE_tip_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0.5, 1., 1.])), plot=False, extrema=True)

                self._TE_base_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0., 0., 0.])), plot=False, extrema=True)
                self._TE_tip_point = geometry.project(ffd_block.evaluate(parametric_coordinates=np.array([0.5, 1., 0.])), plot=False, extrema=True)
                t6 = time.time()

                self._ffd_block = self._make_ffd_block(self.geometry, tight_fit=False)

                # print("time for computing corner points", t6-t5)
            # internal geometry projection info
            self._dependent_geometry_points = [] # {'parametric_points', 'function_space', 'fitting_coords', 'mirror'}
            self._base_geometry = self.geometry.copy()
    
    def create_internal_geometry(
            self,
            top_geometry:lg.Geometry,
            bottom_geometry:lg.Geometry,
            spar_locations:np.ndarray=None,
            spar_termination:int=None,
            spar_function_space:lfs.FunctionSpace=None,
            surf_index:int=1000,
            offset:np.ndarray=np.array([0., 0., .23])
        ):
        '''
        Constructs the internal geometries (surfaces) of the rotor blade from:
            spar_locations: number of spar surfaces and percent of chord (0= le, 1 = te)
            top skin surface geometry
            bottom skin surface geomtry
        '''
        csdl.check_parameter(spar_locations, "spar_locations", types=np.ndarray, allow_none=True)
        csdl.check_parameter(surf_index, "surf_index", types=int)
        
        #TODO: add a parameter to specify at what spanwise coordinate the spar should terminate (usually something like 85-90% of span)

        # Check if spar and rib locations are between 0 and 1
        if spar_locations is not None:
            if not np.all((spar_locations > 0) & (spar_locations < 1)):
                raise ValueError("all spar locations must be between 0 and 1 (excluding the endpoints)")
        if spar_locations is None:
            spar_locations = np.array([0.25, 0.75])
        
        blade = self
        self.spar_termination = spar_termination

        #get and store indices of surfaces in blade
        top_index =list(top_geometry.function_names.keys())[0]
        bot_index =list(bottom_geometry.function_names.keys())[0]
        front_spar_index = surf_index
        rear_spar_index = surf_index+1
        
        num_spars = spar_locations.shape[0]
        if num_spars != 2:
            raise Exception("Provide exactly two normalized spar locations.")
        
        num_spanwise = 10
        if spar_function_space is None:
            spar_function_space = lfs.BSplineSpace(2, (3, 1), (num_spanwise, 2))

        #TODO: need to be able to determine where the root and where the tip is...
        # gather important points 
        root_te = blade.geometry.evaluate(blade._TE_tip_point, non_csdl=True)
        root_le = blade.geometry.evaluate(blade._LE_tip_point, non_csdl=True)
        r_tip_te = blade.geometry.evaluate(blade._TE_base_point, non_csdl=True)
        r_tip_le = blade.geometry.evaluate(blade._LE_base_point, non_csdl=True)
        
        # get spar start/end points (root and tip)
        root_tip_pts = np.zeros((num_spars, 2, 3))
        for i in range(num_spars):
            root_tip_pts[i,0] = (1-spar_locations[i]) * root_le + spar_locations[i] * root_te
            root_tip_pts[i,1] = (1-spar_locations[i]) * r_tip_le + spar_locations[i] * r_tip_te

        # #compute projection direction (perpendicular to the chordline)
        # chord_direction_root = root_le-root_te
        # chord_direction_tip = r_tip_le-r_tip_te
        spanwise_direction = np.array([0,1,0])
        
        LE_base_coord = self.geometry.evaluate(self._LE_base_point).value
        LE_tip_coord = self.geometry.evaluate(self._LE_tip_point).value
        self.blade_length = LE_tip_coord[1]-LE_base_coord[1]
        spar_start = LE_base_coord
        spar_end = LE_tip_coord
        spar_end[1] = LE_tip_coord[1]-(1-spar_termination)*self.blade_length
        self.spar_length = spar_end[1]-spar_start[1]
        u_coords_spar=self._get_parametric_spacing(top_geometry,spar_start,spar_end,num_spanwise)

        parametric_LE = np.vstack([u_coords_spar,np.ones((num_spanwise))]).T
        parametric_TE = np.vstack([u_coords_spar,np.zeros((num_spanwise))]).T

        parametric_LE_tuple_list = [(top_index,parametric_LE[i,:]) for i in range(parametric_LE.shape[0])]
        parametric_TE_tuple_list = [(top_index,parametric_TE[i,:]) for i in range(parametric_TE.shape[0])]
        #minor bug in .evaluate() 
        # physical_LE = blade.geometry.evaluate([(top_index,parametric_LE)],non_csdl=True)
        # physical_TE = blade.geometry.evaluate([(top_index,parametric_TE)],non_csdl=True)
        physical_LE = blade.geometry.evaluate(parametric_LE_tuple_list,non_csdl=True)
        physical_TE = blade.geometry.evaluate(parametric_TE_tuple_list,non_csdl=True)

        #determine chord direction for each section
        chord_direction = physical_LE-physical_TE
        #determine direction perpendicular to chord direction in section plane
        proj_direction = np.cross(chord_direction,np.tile(spanwise_direction,(num_spanwise,1)))

        fwd_spar_chord_loc = (1-spar_locations[0]) * physical_LE + spar_locations[0] * physical_TE
        rear_spar_chord_loc = (1-spar_locations[1]) * physical_LE + spar_locations[1] * physical_TE

        # get parametric coordinates for spars       
        spar_top_front = np.empty((num_spanwise),dtype='O,O')
        spar_bot_front = np.empty((num_spanwise),dtype='O,O')
        spar_top_rear = np.empty((num_spanwise),dtype='O,O')
        spar_bot_rear = np.empty((num_spanwise),dtype='O,O')
        for i in range(num_spanwise):

            spar_top_front[i] = top_geometry.project(fwd_spar_chord_loc[i,:], direction=proj_direction[i,:], grid_search_density_parameter=10)[0]
            spar_top_rear[i] = top_geometry.project(rear_spar_chord_loc[i,:], direction=proj_direction[i,:], grid_search_density_parameter=10)[0]
            spar_bot_rear [i]= bottom_geometry.project(rear_spar_chord_loc[i,:], direction=-proj_direction[i,:], grid_search_density_parameter=10)[0]
            spar_bot_front[i] = bottom_geometry.project(fwd_spar_chord_loc[i,:], direction=-proj_direction[i,:], grid_search_density_parameter=10)[0]

        #create spar surfaces

        #front spar internal surface
        parametric_points_front = spar_top_front.tolist()+spar_bot_front.tolist()
        u_coords=np.linspace(0,1,num_spanwise)
        fitting_coords = np.array([[u, 0] for u in u_coords] + [[u, 1] for u in u_coords])
        fitting_values = self.geometry.evaluate(parametric_points_front)
        coefficients = spar_function_space.fit(fitting_values, fitting_coords)
        front_spar = lfs.Function(spar_function_space, coefficients)
        front_spar_name="Blade_spar_front"+str(front_spar_index)
        self._add_geometry(front_spar_index, front_spar, front_spar_name)

        #rear spar internal surface
        # parametric_points_rear = spar_top_rear.tolist()+spar_bot_rear.tolist()
        parametric_points_rear = spar_bot_rear.tolist()+spar_top_rear.tolist() #fit surface the other direction to ensure normal offset direction offsets spar box inward
        fitting_coords = np.array([[u, 0] for u in u_coords] + [[u, 1] for u in u_coords])
        fitting_values = self.geometry.evaluate(parametric_points_rear)
        coefficients = spar_function_space.fit(fitting_values, fitting_coords)
        rear_spar = lfs.Function(spar_function_space, coefficients)
        rear_spar_name = "Blade_spar_rear"+str(rear_spar_index)
        self._add_geometry(rear_spar_index, rear_spar, rear_spar_name)

        front_spar_geometry = self.create_subgeometry(search_names=front_spar_name)
        rear_spar_geometry = self.create_subgeometry(search_names=rear_spar_name)

        return front_spar_geometry,rear_spar_geometry
    
    def create_beam_xs_meshes(
            self,
            top_geometry:lg.Geometry,
            bottom_geometry:lg.Geometry,
            front_spar_geometry:lg.Geometry=None,
            rear_spar_geometry:lg.Geometry=None,
            num_spanwise=10,
            xs_surf_index=2000
        ):

        self.xs_surf_index = xs_surf_index
        #get surface indices
        top_index =list(top_geometry.function_names.keys())[0]
        bot_index =list(bottom_geometry.function_names.keys())[0]
        if front_spar_geometry is not None:
            front_spar_index =list(front_spar_geometry.function_names.keys())[0]
        if rear_spar_geometry is not None:
            rear_spar_index =list(rear_spar_geometry.function_names.keys())[0]
        
        #number of parametric points to be evaluated for each surface
        num_parametric = 500
        # u_coords = np.linspace(0,1,num_spanwise)
        #ensure even spacing in physical space
        LE_base_coord = self.geometry.evaluate(self._LE_base_point).value
        LE_tip_coord = self.geometry.evaluate(self._LE_tip_point).value
        u_coords = self._get_parametric_spacing(top_geometry,LE_base_coord,LE_tip_coord,num_spanwise)
        #TODO:insert the endpoint of the spar box into the u_coords
        u_coord_spar_termination=top_geometry.project(front_spar_geometry.evaluate([(front_spar_index, np.array([1.0,1.0]))]),direction=[1,0,1])[0][1][0][0]
        u_coords_insert=(u_coords>u_coord_spar_termination).nonzero()[0][0]
        u_coords_new = np.insert(u_coords,u_coords_insert,u_coord_spar_termination)
        v_coords = np.linspace(0,1,num_parametric) # linear spacing doesn't perform well

        #TODO: use a coordinate transform to ensure that the meshing is always handled in xy-plane,
        #   then re-map back to physical space after computing the section meshes?
        #   Currently, this meshing only works if the rotor blade has it's beam axis directly along the y-axis

        xs = []
        
        # for i,u_coord in enumerate(u_coords):
        start_xs = 5            
        for i,u_coord in zip([i+start_xs for i in range(len(u_coords[start_xs:]))],u_coords[start_xs:]):            
            #NEED TO USE A CONSTANT AXIAL COORDINATE TO PREVENT MESHING ISSUES ASSOCIATED WITH FITTING PLANAR SURFACES
            parametric_top = [(top_index, np.array([u_coord,v_coord])) for v_coord in v_coords]
            axial_coord = np.average(self.geometry.evaluate(parametric_top).value[:,1])

            #========== SKIN CONSTRUCTION ==========#
            #TOP SURFACE:
            print(f'Section {i}/{num_spanwise}')
            parametric_top = [(top_index, np.array([u_coord,v_coord])) for v_coord in v_coords]
            #return pts and valid offset points (along with valid offset point indices)
            (top_pts,
                top_pts_offset,
                valid_top_offset_pts_indices) = self._get_pts_and_offset_pts(parametric_top,return_indices=True) 
            # #TODO: in an optimization run, swap for the design variable upper limit:
            top_thicknesses= self._get_thicknesses(parametric_top,0).value
            print("num top pts="+str(top_pts.shape[0])+', num offset pts:'+str(top_pts_offset.shape[0]))

            #BOTTOM SURFACE:
            parametric_bot = [(bot_index, np.array([u_coord, v_coord])) for v_coord in v_coords]
            (bot_pts,
                bot_pts_offset,
                valid_bot_offset_pts_indices) = self._get_pts_and_offset_pts(parametric_bot,return_indices=True) 
            bot_thicknesses=self._get_thicknesses(parametric_bot,0).value
            print("num bot pts="+str(bot_pts.shape[0])+', num offset pts:'+str(bot_pts_offset.shape[0]))
            
            #combine skin surface points into a single array starting at the trailing edge on the upper surface
            if np.isclose(bot_pts[0,:].value, top_pts[-1,:].value).all():
                skin_pts = csdl.concatenate([top_pts,bot_pts[1:,:]])
            else:
                skin_pts = csdl.concatenate([top_pts,bot_pts])

            #for the offset pts need to detect if there is an intersection between the upper and lower curves
            # since we used the ROC constraint, we know there can only be a maximum of two intersections 
            # from the discontinuous derivative at the leading/trailing edges
            # additionally, since the same number of parametric points are used to construct each curve, we can 
            # detect the le vs te based on comparing which index is greater 
            #   upper_index < lower_index = te
            #   upper_index > lower_index = le

            intersections = self._find_pt_set_intersection(top_pts_offset,bot_pts_offset)
            #depending on the number of intersections, modify the collection of offset points
            if intersections is None:    
                print("No intersections detected")
            elif len(intersections)==1:        
                #get intersetion point:
                intersection_pt = np.array([[intersections[0][2][0],
                                            top_pts_offset[0,1].value[0],
                                            intersections[0][2][1]]])
                #detect whether it was the leading or trailing edge with an intersection
                #trailing edge case:
                if intersections[0][0]<intersections[0][1]:
                    print("One intersection, trailing edge") 
                    #trim offset points before and after intersection:
                    top_pts_offset = top_pts_offset[(intersections[0][0]+1):,:]
                    bot_pts_offset = bot_pts_offset[:intersections[0][1],:]
                    #update the valid offsets indices:
                    valid_top_offset_pts_indices = valid_top_offset_pts_indices[(intersections[0][0]+1):]
                    #add intersection point to upper and lower skin (close loop)
                    top_pts_offset = csdl.concatenate([intersection_pt,top_pts_offset])
                    bot_pts_offset = csdl.concatenate([bot_pts_offset,intersection_pt])
                    #update valid offsets:
                    valid_top_offset_pts_indices = np.insert(valid_top_offset_pts_indices, 0, -1, axis=0)

                #leading edge case:
                else:
                    print("One intersection, leading edge") 
                    #trim offset points before and after intersection:
                    top_pts_offset = top_pts_offset[:intersections[0][0],:]
                    bot_pts_offset = bot_pts_offset[intersections[0][1]+1,:]
                    #add intersection point (only to top skin offset):
                    top_pts_offset = csdl.concatenate([top_pts_offset,intersection_pt])

            elif len(intersections)==2:
                print("Two intersections, leading & trailing edges:")
                #get intersection points:
                y_value = top_pts_offset[0,1].value[0]
                #determine which intersection is leading vs trailing edge:
                if intersections[0][0]<intersections[0][1]:
                    te_intersection = intersections[0]
                    le_intersection = intersections[1]
                else:
                    te_intersection = intersections[1]
                    le_intersection = intersections[0]
                te_intersection_pt = np.array([[te_intersection[2][0],
                                            y_value,
                                            te_intersection[2][1]]])
                le_intersection_pt = np.array([[le_intersection[2][0],
                                            y_value,
                                            le_intersection[2][1]]])                
                #trim offset points:
                top_pts_offset = top_pts_offset[te_intersection[0]+1:le_intersection[0],:]
                bot_pts_offset = bot_pts_offset[le_intersection[1]+1:te_intersection[1],:]
                #add intersection points:
                top_pts_offset = csdl.concatenate([te_intersection_pt,top_pts_offset,le_intersection_pt])
                bot_pts_offset = csdl.concatenate([bot_pts_offset,te_intersection_pt])
            else:
                raise ValueError("Incorrect number of intersections detected")

            #remove duplicate points from leading edge, if present
            if np.isclose(bot_pts_offset[0,:].value, top_pts_offset[-1,:].value).all():
                bot_pts_offset=bot_pts_offset[1:,:]
            #ensure that the trailing edge point is duplicated
            if not np.isclose(bot_pts_offset[-1,:].value, top_pts_offset[0,:].value).all():
                raise ValueError("Trailing edge points do not match")
            
            #join the two collections of offset points:
            skin_offset_pts = csdl.concatenate([top_pts_offset,bot_pts_offset])
            
            print("num skin pts="+str(skin_pts.shape[0])+', num offset pts:'+str(skin_offset_pts.shape[0]))
            
            n_thickness = 3
            approx_mesh_size_skin=np.min(np.vstack([top_thicknesses,bot_thicknesses]))/n_thickness

            #fit a single surface to the outer skin
            skin_surf_name='skin_'+str(i)
            # skin_surf_geometry = self._fit_xs_surface(skin_pts,
            #                                         skin_offset_pts,
            #                                         skin_surf_name,
            #                                         num_parametric)
            # skin_surf_geometry.plot(opacity=0.75)
            skin_mesh = self._mesh_curve_and_offset(skin_pts,
                                                    skin_offset_pts,
                                                    name=skin_surf_name,
                                                    plot=False,
                                                    meshsize=approx_mesh_size_skin,
                                                    num_boundary_comp=2)
            # skin_output = cd.mesh_utils.import_mesh(file=skin_mesh,
            #                           component=skin_surf_geometry,
            #                           plot=True)
            
            #CHECK IF THE SPANWISE COORDINATE IS BEFORE OR AFTER THE SPAR TERMINATION:
            front_spar_end=front_spar_geometry.evaluate([(front_spar_index, np.array([1.0,1.0]))])
            if front_spar_end.value[1]>=axial_coord:
                print('still got it :)')
            #========== FRONT SPAR CONSTRUCTION ==========#
            #TODO: need to get the correct parametric u_coord from the parametric u_coord defined on the skin
            #   First, need to check whether the spar box as terminated or not
            #   If, so do not try to mesh the front spar
            # front_spar_geometry.project(top_geometry.evaluate([(top_index, np.array([u_coord,1])) for u_coord in u_coords[:8]]),direction=[0,1,0],plot=True)
            u_coord_front_spar=[parametric[1][0][0] 
                                for parametric in front_spar_geometry.project(
                                                    top_geometry.evaluate([(top_index, np.array([u_coord,1]))]),
                                                                           direction=[1,0,1])][0]
            parametric_front_spar = [(front_spar_index, np.array([u_coord_front_spar,v_coord])) for v_coord in v_coords]
            front_spar_pts,front_spar_pts_offset = self._get_pts_and_offset_pts(parametric_front_spar)
            #trim points off the spar curves that intersect with the inner edge of the skin
            #   and add the inner surface curve points to construct the edge that is "shared" with the skin
            #   The spar curve itself will always have 2 intersections
            #   Can have cases with the offset spar curve and skin offset have 0,1,or 2 intersections
            
            #this is guaranteed to have exactly 2 intersections (based on how it was constructed)
            front_spar_skin_intersections=self._find_pt_set_intersection(front_spar_pts,skin_offset_pts)
            front_spar_indices = np.sort([intersection[0] for intersection in front_spar_skin_intersections])
            front_spar_pts = front_spar_pts[front_spar_indices[0]+1:front_spar_indices[1]]

            #add the intersection point to the front spar points:
            y_value = front_spar_pts[0,1].value[0]
            top_intersection_pt_front = np.array([[front_spar_skin_intersections[0][2][0],
                                            y_value,
                                            front_spar_skin_intersections[0][2][1]]])
            bot_intersection_pt_front = np.array([[front_spar_skin_intersections[1][2][0],
                                            y_value,
                                            front_spar_skin_intersections[1][2][1]]])
            front_spar_pts = csdl.concatenate([top_intersection_pt_front,
                                               front_spar_pts,
                                               bot_intersection_pt_front])

            # store the intersections with the skin for trimming the curve
            skin_indices_front = np.sort([intersection[1] for intersection in front_spar_skin_intersections])
            
            # either 0,1,or 2interesections
            front_spar_offset_skin_intersections=self._find_pt_set_intersection(front_spar_pts_offset,skin_offset_pts)
            front_spar_thicknesses=self.quantities.material_properties.evaluate_thickness(parametric_front_spar).value
            
            #if there are not two intersections, locate which ends of the spar offset curve are NOT intersecting,
            #  then extend them with a straight line in the direction of the last segment
            if front_spar_offset_skin_intersections is None:
                print("no intersections!")
                #extend the spar offset points in the tangent direction of the offset
                #  points by using the wingskin surface tangent 

                #evaluate normals and thicknesses for spar points
                spar_skin_pts = [parametric_front_spar[0],parametric_front_spar[-1]]
                front_spar_ext_thickness=self.quantities.material_properties.evaluate_thickness(spar_skin_pts)
                front_spar_ext_normal = self.geometry.evaluate_normals(spar_skin_pts)
                front_spar_ext_normal_inplane = rotate_oblique(front_spar_ext_normal)
                
                #get extension points
                #===== UPPER EDGE =======#
                #evaluate front spar point, project to top surface, then evaluate derivative at that location
                tangent_vec_top = top_geometry.evaluate(
                                        top_geometry.project(self.geometry.evaluate([spar_skin_pts[0]])),
                                        parametric_derivative_orders=(0,1))
                tangent_vec_top /= csdl.norm(tangent_vec_top) #normalize
                e_top = get_extension_vector(tangent_vec_top,front_spar_ext_normal_inplane[0])
                ext_pt_top = front_spar_pts_offset[0] + front_spar_ext_thickness[0]*e_top 
                #===== LOWER EDGE =======#
                #evaluate front spar point, project to bottom surface, then evaluate derivative at that location
                tangent_vec_bot = bottom_geometry.evaluate(
                                        bottom_geometry.project(self.geometry.evaluate([spar_skin_pts[1]])),
                                        parametric_derivative_orders=(0,1))
                tangent_vec_bot /= csdl.norm(tangent_vec_bot) #normalize
                e_bot = get_extension_vector(tangent_vec_bot,front_spar_ext_normal_inplane[1])
                ext_pt_bot = front_spar_pts_offset[-1] + front_spar_ext_thickness[1]*e_bot

                #add the extended points to the offset points array
                front_spar_pts_offset = csdl.concatenate([ext_pt_top.reshape((1,3)),
                                                          front_spar_pts_offset,
                                                          ext_pt_bot.reshape((1,3))])
                #find the intersection between that and the skin offset pts:
                front_spar_offset_skin_intersections=self._find_pt_set_intersection(front_spar_pts_offset,skin_offset_pts)


            elif len(front_spar_offset_skin_intersections) == 1:
                print("one intersection! ")

                #identify which end point is lacking an intersection
                #upper edge case
                if front_spar_offset_skin_intersections[0][0]>=front_spar_pts_offset.shape[0]/2:
                    #evaluate normals and thicknesses for spar points
                    spar_skin_pts = [parametric_front_spar[0],parametric_front_spar[0]]
                    front_spar_ext_thickness=self.quantities.material_properties.evaluate_thickness(spar_skin_pts)
                    front_spar_ext_normal = self.geometry.evaluate_normals(spar_skin_pts)
                    front_spar_ext_normal_inplane = rotate_oblique(front_spar_ext_normal)
                    #evaluate front spar point, project to top surface, then evaluate derivative at that location
                    tangent_vec_top = top_geometry.evaluate(
                                            top_geometry.project(self.geometry.evaluate([spar_skin_pts[0]])),
                                            parametric_derivative_orders=(0,1))
                    tangent_vec_top /= csdl.norm(tangent_vec_top) #normalize
                    e_top = get_extension_vector(tangent_vec_top,front_spar_ext_normal_inplane[0])
                    ext_pt_top = front_spar_pts_offset[0] + front_spar_ext_thickness[0]*e_top 
                    
                    #add the extended points to the offset points array
                    front_spar_pts_offset = csdl.concatenate([ext_pt_top.reshape((1,3)),
                                                            front_spar_pts_offset])
                    #find the intersection between that and the skin offset pts:
                    front_spar_offset_skin_intersections=self._find_pt_set_intersection(front_spar_pts_offset,skin_offset_pts)
                
                #lower edge case
                elif front_spar_offset_skin_intersections[0][0]<front_spar_pts_offset.shape[0]/2:
                    #evaluate normals and thicknesses for spar points
                    #REPEAT TO PREVENT THROWING ERROR:
                    spar_skin_pts = [parametric_front_spar[-1],parametric_front_spar[-1]]
                    front_spar_ext_thickness=self.quantities.material_properties.evaluate_thickness(spar_skin_pts)
                    front_spar_ext_normal = self.geometry.evaluate_normals(spar_skin_pts)
                    front_spar_ext_normal_inplane = rotate_oblique(front_spar_ext_normal)
                    #evaluate front spar point, project to bottom surface, then evaluate derivative at that location
                    tangent_vec_bot = bottom_geometry.evaluate(
                                            bottom_geometry.project(self.geometry.evaluate([spar_skin_pts[0]])),
                                            parametric_derivative_orders=(0,1))
                    tangent_vec_bot /= csdl.norm(tangent_vec_bot) #normalize
                    e_bot = get_extension_vector(tangent_vec_bot,front_spar_ext_normal_inplane[1])
                    ext_pt_bot = front_spar_pts_offset[-1] + front_spar_ext_thickness[0]*e_bot

                    #add the extended points to the offset points array
                    front_spar_pts_offset = csdl.concatenate([front_spar_pts_offset,
                                                            ext_pt_bot.reshape((1,3))])
                    #find the intersection between that and the skin offset pts:
                    front_spar_offset_skin_intersections=self._find_pt_set_intersection(front_spar_pts_offset,skin_offset_pts)
                else:
                    print("could not correctly identify the missing intersection end :(")
                    
            elif len(front_spar_offset_skin_intersections) == 2:
                print("two intersections!")
                #No need to do anything here, carry on

            #modify the offset spar points to include the intersection point between that curve and the skin offset curve
            top_intersection_pt_front_offset = np.array([[front_spar_offset_skin_intersections[0][2][0],
                                                            y_value,
                                                            front_spar_offset_skin_intersections[0][2][1]]])
            bot_intersection_pt_front_offset = np.array([[front_spar_offset_skin_intersections[1][2][0],
                                        y_value,
                                        front_spar_offset_skin_intersections[1][2][1]]])
            #get new spar offset points:
            front_spar_pts_offset = csdl.concatenate([top_intersection_pt_front_offset,
                                                        front_spar_pts_offset[front_spar_offset_skin_intersections[0][0]+1:front_spar_offset_skin_intersections[1][0]],
                                                        bot_intersection_pt_front_offset])
            #indices of the offset spar 
            skin_indices_front_offset = np.sort([intersection[1] for intersection in front_spar_offset_skin_intersections])
            print(f'front skin indices:{skin_indices_front[0]},{skin_indices_front[1]}')

            #pts for the upper spar curve:
            front_spar_top_pts = csdl.concatenate([top_intersection_pt_front_offset,
                                                    skin_offset_pts[skin_indices_front_offset[0]+1:skin_indices_front[0]],
                                                    top_intersection_pt_front])
            #pts for the lower spar curve:
            front_spar_bot_pts = csdl.concatenate([bot_intersection_pt_front,
                                                    skin_offset_pts[skin_indices_front[1]+1:skin_indices_front_offset[1]],
                                                    bot_intersection_pt_front_offset])

            #Set a consistent value for all spar surface curves:
            def set_axial_value(vec,val):
                vec.set_value(np.vstack([vec[:,0].value,
                                            np.ones_like(vec[:,1].value)*val,
                                            vec[:,2].value]).T)
                return
            set_axial_value(front_spar_top_pts,axial_coord)
            set_axial_value(front_spar_bot_pts,axial_coord)            
            set_axial_value(front_spar_pts,axial_coord)            
            set_axial_value(front_spar_pts_offset,axial_coord)            

            n_thickness_front_spar = 5
            approx_mesh_size_front_spar=np.min(front_spar_thicknesses)/n_thickness_front_spar

            
            front_spar_surf_name='front_spar_'+str(i)
            #TODO: need new surface fitting to account for the curve at the top and bottom skin
            # front_spar_surf_geometry = self._fit_xs_surface(front_spar_pts,
            #                                         front_spar_pts_offset,
            #                                         front_spar_surf_name,
            #                                         num_parametric)
            # front_spar_surf_geometry.plot(opacity=0.75)

            front_spar_mesh = self._mesh_curve_loop([front_spar_pts,
                                                     front_spar_bot_pts,#provided ordered from front spar --> offset
                                                     front_spar_pts_offset[::-1],#reverse to ensure proper curve loop direction
                                                     front_spar_top_pts],#provided ordered from front spar --> offset
                                                     name=front_spar_surf_name,
                                                     plot=False,
                                                     meshsize=approx_mesh_size_front_spar)
            # front_spar_mesh = self._mesh_curve_and_offset(front_spar_pts,
            #                                         front_spar_pts_offset,
            #                                         name=front_spar_surf_name,
            #                                         plot=True,
            #                                         meshsize=approx_mesh_size_front_spar)
            # front_spar_output = cd.mesh_utils.import_mesh(file=front_spar_mesh,
            #                           component=front_spar_surf_geometry,
            #                           plot=True)
            
            #========== REAR SPAR CONSTRUCTION ==========#
            u_coord_rear_spar=[parametric[1][0][0] 
                                for parametric in rear_spar_geometry.project(
                                                    top_geometry.evaluate([(top_index, np.array([u_coord,1]))]),
                                                                           direction=[1,0,1])][0]
            parametric_rear_spar = [(rear_spar_index, np.array([u_coord_rear_spar,v_coord])) for v_coord in v_coords]
            rear_spar_pts,rear_spar_pts_offset = self._get_pts_and_offset_pts(parametric_rear_spar)
            #trim points off the spar curves that intersect with the inner edge of the skin
            #   and add the inner surface curve points to construct the edge that is "shared" with the skin
            #   The spar curve itself will always have 2 interesections
            #   Can have cases with the offset spar curve and skin offset have 0,1,or 2 intersections
            
            #this is guaranteed to have exactly 2 intersections (based on how it was constructed)
            #TODO: makes sure to order 
            rear_spar_skin_intersections=self._find_pt_set_intersection(rear_spar_pts,skin_offset_pts)
            rear_spar_indices = np.sort([intersection[0] for intersection in rear_spar_skin_intersections])
            rear_spar_pts = rear_spar_pts[rear_spar_indices[0]+1:rear_spar_indices[1]]

            #add the intersection point to the rear spar points:
            y_value = rear_spar_pts[0,1].value[0]
            top_intersection_pt_rear = np.array([[rear_spar_skin_intersections[1][2][0],
                                            y_value,
                                            rear_spar_skin_intersections[1][2][1]]])
            bot_intersection_pt_rear = np.array([[rear_spar_skin_intersections[0][2][0],
                                            y_value,
                                            rear_spar_skin_intersections[0][2][1]]])
            rear_spar_pts = csdl.concatenate([bot_intersection_pt_rear,
                                               rear_spar_pts,
                                               top_intersection_pt_rear])

            # store the intersections with the skin for trimming the curve
            skin_indices_rear = np.sort([intersection[1] for intersection in rear_spar_skin_intersections])
            print(f'rear skin indices:{skin_indices_rear[0]},{skin_indices_rear[1]}')
            # either 0,1,or 2interesections
            rear_spar_offset_skin_intersections=self._find_pt_set_intersection(rear_spar_pts_offset,skin_offset_pts)
            rear_spar_thicknesses=self.quantities.material_properties.evaluate_thickness(parametric_rear_spar).value
            
            #if there are not two intersections, locate which ends of the spar offset curve are NOT intersecting,
            #  then extend them with a straight line in the direction of the last segment
            if rear_spar_offset_skin_intersections is None:
                print("no intersections!")
                #extend the spar offset points in the tangent direction of the offset
                #  points by using the wingskin surface tangent 

                #evaluate normals and thicknesses for spar points
                spar_skin_pts = [parametric_rear_spar[0],parametric_rear_spar[-1]]
                rear_spar_ext_thickness=self.quantities.material_properties.evaluate_thickness(spar_skin_pts)
                rear_spar_ext_normal = self.geometry.evaluate_normals(spar_skin_pts)
                
                #evaluate rear spar point, project to top surface, then evaluate derivative at that location
                tangent_vec = top_geometry.evaluate(
                                top_geometry.project(
                                        self.geometry.evaluate(spar_skin_pts)
                                            ),parametric_derivative_orders=(0,1))
                tangent_vec /= csdl.norm(tangent_vec) #normalize

                rear_spar_ext_normal_inplane = rotate_oblique(rear_spar_ext_normal)

                #get extension points
                #===== UPPER EDGE =======#
                e_top = get_extension_vector(tangent_vec[0],rear_spar_ext_normal_inplane[0])
                ext_pt_top = rear_spar_pts_offset[0] + rear_spar_ext_thickness[0]*e_top 
                #===== LOWER EDGE =======#
                e_bot = get_extension_vector(tangent_vec[1],rear_spar_ext_normal_inplane[1])
                ext_pt_bot = rear_spar_pts_offset[1] + rear_spar_ext_thickness[1]*e_bot

                #add the extended points to the offset points array
                rear_spar_pts_offset = csdl.concatenate([ext_pt_top.reshape((1,3)),
                                                          rear_spar_pts_offset,
                                                          ext_pt_bot.reshape((1,3))])
                #find the intersection between that and the skin offset pts:
                rear_spar_offset_skin_intersections=self._find_pt_set_intersection(rear_spar_pts_offset,skin_offset_pts)


            elif len(rear_spar_offset_skin_intersections) == 1:
                #TODO: detect which section doesn't have an intersection
                print("one intersection! ")

                #identify which end point is lacking an intersection

                #extend the offset curve in that direction

                #recompute the offset curve intersections (especially if the missing intersection was on the top, as that means subsequent indices would need to get adjusted)

            elif len(rear_spar_offset_skin_intersections) == 2:
                print("two intersections!")
                #No need to do anything here, carry on

            #modify the offset spar points to include the intersection point between that curve and the skin offset curve
            top_intersection_pt_rear_offset = np.array([[rear_spar_offset_skin_intersections[0][2][0],
                                                            y_value,
                                                            rear_spar_offset_skin_intersections[0][2][1]]])
            bot_intersection_pt_rear_offset = np.array([[rear_spar_offset_skin_intersections[1][2][0],
                                        y_value,
                                        rear_spar_offset_skin_intersections[1][2][1]]])
            #get new spar offset points:
            rear_spar_pts_offset = csdl.concatenate([top_intersection_pt_rear_offset,
                                                        rear_spar_pts_offset[rear_spar_offset_skin_intersections[0][0]+1:rear_spar_offset_skin_intersections[1][0]],
                                                        bot_intersection_pt_rear_offset])
            #indices of the offset spar 
            skin_indices_rear_offset = np.sort([intersection[1] for intersection in rear_spar_offset_skin_intersections])
            
            #pts for the upper spar curve:
            rear_spar_top_pts = csdl.concatenate([top_intersection_pt_rear_offset,
                                                    skin_offset_pts[skin_indices_rear_offset[0]+1:skin_indices_rear[0]],
                                                    top_intersection_pt_rear])
            #pts for the lower spar curve:
            rear_spar_bot_pts = csdl.concatenate([bot_intersection_pt_rear,
                                                    skin_offset_pts[skin_indices_rear[1]+1:skin_indices_rear_offset[1]],
                                                    bot_intersection_pt_rear_offset])

            #set a consistent axial coordinate
            set_axial_value(rear_spar_top_pts,axial_coord)
            set_axial_value(rear_spar_bot_pts,axial_coord)            
            set_axial_value(rear_spar_pts,axial_coord)            
            set_axial_value(rear_spar_pts_offset,axial_coord)            

            n_thickness_rear_spar = 5
            approx_mesh_size_rear_spar=np.min(rear_spar_thicknesses)/n_thickness_rear_spar
            
            rear_spar_surf_name='rear_spar_'+str(i)
            #TODO: need new surface fitting to account for the curve at the top and bottom skin
            # rear_spar_surf_geometry = self._fit_xs_surface(rear_spar_pts,
            #                                         rear_spar_pts_offset,
            #                                         rear_spar_surf_name,
            #                                         num_parametric)
            # rear_spar_surf_geometry.plot(opacity=0.75)

            rear_spar_mesh = self._mesh_curve_loop([rear_spar_pts,
                                                     rear_spar_bot_pts,#provided ordered from rear spar --> offset
                                                     rear_spar_pts_offset[::-1],#reverse to ensure proper curve loop direction
                                                     rear_spar_top_pts],#provided ordered from rear spar --> offset
                                                     name=rear_spar_surf_name,
                                                     plot=False,
                                                     meshsize=approx_mesh_size_rear_spar)
            # rear_spar_mesh = self._mesh_curve_and_offset(rear_spar_pts,
            #                                         rear_spar_pts_offset,
            #                                         name=rear_spar_surf_name,
            #                                         plot=True,
            #                                         meshsize=approx_mesh_size_rear_spar)
            # rear_spar_output = cd.mesh_utils.import_mesh(file=rear_spar_mesh,
            #                           component=rear_spar_surf_geometry,
            #                           plot=True)
            
            #========== TOP SPAR CAP CONSTRUCTION ==========#
            #TODO: indices for the offset have some issues (need to dig into this a bit more)
            top_spar_pts = csdl.concatenate([top_intersection_pt_rear,
                                            skin_offset_pts[skin_indices_rear[0]+1:skin_indices_front[0]],
                                            top_intersection_pt_front])
            
            #find the parametric points corresponding to the valid offset points between the spar surfaces
            top_to_top_spar_offset_pts_indices = valid_top_offset_pts_indices[skin_indices_rear[0]+1:skin_indices_front[0]]
            parametric_top_spar_valid_offset = [parametric_top[indx] for indx in top_to_top_spar_offset_pts_indices]
            (top_spar_pts_backup,
                top_spar_pts_offset,
                valid_top_spar_offset_pts_indices) = self._get_pts_and_offset_pts(parametric_top_spar_valid_offset,
                                                                      layer='all',
                                                                      return_indices=True) 

            top_spar_thicknesses= self._get_thicknesses(parametric_top,1).value
            
            n_thickness_top_spar = 4
            approx_mesh_size_top_spar=np.min(top_spar_thicknesses)/n_thickness_top_spar

            top_spar_surf_name='top_spar_'+str(i)
            
            #verify the axial coordinate is set properly
            set_axial_value(top_spar_pts,axial_coord)
            set_axial_value(top_spar_pts_offset,axial_coord)

            self._mesh_curve_and_offset(top_spar_pts,
                                            top_spar_pts_offset,
                                            name=top_spar_surf_name,
                                            plot=False,
                                            meshsize=approx_mesh_size_top_spar)

            #TODO: need to also project onto skin            
            
            #========== BOTTOM SPAR CAP CONSTRUCTION ==========#
            bot_spar_pts = csdl.concatenate([bot_intersection_pt_front,
                                            skin_offset_pts[skin_indices_front[1]+1:skin_indices_rear[1]],
                                            bot_intersection_pt_rear])
            
            #find the parametric points corresponding to the valid offset points between the spar surfaces
            num_top_offset_pts = top_pts_offset.shape[0]
            start = (skin_indices_front[1]-num_top_offset_pts)+1
            end = skin_indices_rear[1]-num_top_offset_pts
            bot_to_bot_spar_offset_pts_indices = valid_bot_offset_pts_indices[start:end]
            parametric_bot_spar_valid_offset = [parametric_bot[indx] for indx in bot_to_bot_spar_offset_pts_indices]
            (bot_spar_pts_backup,
                bot_spar_pts_offset,
                valid_bot_spar_offset_pts_indices) = self._get_pts_and_offset_pts(parametric_bot_spar_valid_offset,
                                                                      layer='all',
                                                                      return_indices=True) 

            bot_spar_thicknesses= self._get_thicknesses(parametric_bot,1).value
            
            n_thickness_bot_spar = 4
            approx_mesh_size_bot_spar=np.min(bot_spar_thicknesses)/n_thickness_bot_spar

            bot_spar_surf_name='bot_spar_'+str(i)
            
            #verify the axial coordinate is set properly
            set_axial_value(bot_spar_pts,axial_coord)
            set_axial_value(bot_spar_pts_offset,axial_coord)

            self._mesh_curve_and_offset(bot_spar_pts,
                                            bot_spar_pts_offset,
                                            name=bot_spar_surf_name,
                                            plot=False,
                                            meshsize=approx_mesh_size_bot_spar)

            #TODO: need to also project onto skin            

            #========== FRONT FILL CONSTRUCTION ==========#
            front_skin_offset_segment = csdl.concatenate([top_intersection_pt_front,
                                                    skin_offset_pts[skin_indices_front[0]+1:skin_indices_front[1]],
                                                    bot_intersection_pt_front])
            set_axial_value(front_skin_offset_segment,axial_coord) #ensure all axial coordinates match

            front_fill_surf_name = 'front_fill_'+str(i)
            front_fill_mesh = self._mesh_curve_loop([front_skin_offset_segment,
                                                     front_spar_pts[::-1]],
                                                     name=front_fill_surf_name,
                                                     plot=False,
                                                     meshsize=approx_mesh_size_rear_spar)

            #========== REAR FILL CONSTRUCTION ==========#
            rear_skin_offset_upper_segment = csdl.concatenate([skin_offset_pts[:skin_indices_rear[0]],
                                                               top_intersection_pt_rear])
            rear_skin_offset_lower_segment = csdl.concatenate([bot_intersection_pt_rear,
                                                               skin_offset_pts[skin_indices_rear[1]+1:]])
            set_axial_value(rear_skin_offset_upper_segment,axial_coord) #ensure all axial coordinates match
            set_axial_value(rear_skin_offset_lower_segment,axial_coord) #ensure all axial coordinates match

            rear_fill_surf_name = 'rear_fill_'+str(i)
            rear_fill_mesh = self._mesh_curve_loop([rear_skin_offset_upper_segment,
                                                     rear_spar_pts[::-1],
                                                     rear_skin_offset_lower_segment],
                                                     name=rear_fill_surf_name,
                                                     plot=False,
                                                     meshsize=approx_mesh_size_rear_spar)
            
            #TODO: construct balance mass surface?
            
                    
        # self.geometry.plot(opacity=0.5)

    def _get_parametric_spacing(self,geometry,start,end,num_spanwise,axis=1):
        ''' geometry:geometry to project onto
            start: xyz coordinate
            end: xyz coordinate
            num_spanwise: number of points (num_segements=num_spanwise-1)'''        
        # axial_coords=np.linspace(LE_base_coord[1],LE_tip_coord[1]-(1-self.spar_termination)*blade_length,num_spanwise)
        if axis==0:
            axial_coords=np.linspace(start[0],end[0],num_spanwise)
            coords = np.vstack([axial_coords,start[1]*np.ones_like(axial_coords),end[2]*np.ones_like(axial_coords)]).T
        elif axis==1:
            axial_coords=np.linspace(start[1],end[1],num_spanwise)
            coords = np.vstack([start[0]*np.ones_like(axial_coords),axial_coords,end[2]*np.ones_like(axial_coords)]).T
        elif axis==2:
            axial_coords=np.linspace(start[2],end[2],num_spanwise)
            coords = np.vstack([start[0]*np.ones_like(axial_coords),start[1]*np.ones_like(axial_coords),axial_coords]).T
        #get the parametric u coordinate for the projection of the evenly spaced physical points:
        u_coords = np.array([parametric[1][0][0] for parametric in geometry.project(coords)])
        # u_coords=np.linspace(0,1,num_spanwise)
        return u_coords

    def plot_vecs(self,origin, vectors, colors=None):
        """
        Plot 2D vectors and their unit vectors from a common origin.

        Parameters:
        - origin: tuple or np.array of shape (2,)
        - vectors: list of np.array of shape (2,)
        - colors: list of colors for each vector (optional)
        """
        origin = np.array(origin)
        vectors = np.array(vectors)
        n = len(vectors)

        if colors is None:
            colors = ['r'] * n  # default to red

        plt.figure()
        ax = plt.gca()

        for i, v in enumerate(vectors):
            norm = np.linalg.norm(v)
            unit_v = v / norm if norm != 0 else np.zeros_like(v)

            # Plot original vector
            ax.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color=colors[i], width=0.005, label=f'vec {i}')
            label_pos = origin + 0.9 * v
            ax.text(label_pos[0], label_pos[1], f'{norm:.2f}', fontsize=9, ha='center', va='center', color=colors[i])

            # Plot unit vector with dashed arrow
            ax.quiver(*origin, *unit_v, angles='xy', scale_units='xy', scale=1, color=colors[i],
                    linestyle='dashed', alpha=0.5, width=0.003)
            unit_label_pos = origin + 0.9 * unit_v
            ax.text(unit_label_pos[0], unit_label_pos[1], f'û{i}', fontsize=8, ha='center', va='center', color=colors[i], alpha=0.7)

        # Plot settings
        max_len = max(np.linalg.norm(v) for v in vectors)
        buffer = 1.0
        plt.xlim(origin[0] - buffer, origin[0] + max_len + buffer)
        plt.ylim(origin[1] - buffer, origin[1] + max_len + buffer)
        ax.set_aspect('equal')
        plt.grid(True)
        plt.title(f'Vectors and Unit Vectors from {tuple(origin)}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()


    def plot_pts(self,pts,offset_pts):
        plt.figure(figsize=(8, 6))
        plt.plot(pts.value[:, 0], pts.value[:, 2], 'b--o', label='Skin')
        plt.plot(offset_pts.value[:, 0], offset_pts.value[:, 2], 'g--o', label='Offset Skin')
        plt.legend()
        plt.title("polyline representation")
        plt.grid(True)
        plt.show()

    def _find_pt_set_intersection(self,pts1,pts2):
        """ Return the possible two intersections between the offset curves"""

        def line_intersection(p1, p2, p3, p4):
            """Returns intersection point of segments (p1,p2) and (p3,p4), or None."""
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4

            denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
            if abs(denom) < 1e-13:
                return None

            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
            point = np.array([px, py])

            if (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and
                min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)):
                return point
            return None

        def find_all_intersections(poly1, poly2):
            """Returns list of (i, j, intersection_point) for all intersections."""
            intersections = []
            for i in range(len(poly1) - 1):
                p1, p2 = poly1[i], poly1[i + 1]
                for j in range(len(poly2) - 1):
                    q1, q2 = poly2[j], poly2[j + 1]
                    inter = line_intersection(p1, p2, q1, q2)
                    if inter is not None:
                        intersections.append((i, j, inter))
            
            if len(intersections) == 0:
                return None
            
            return intersections
        
        return find_all_intersections(pts1.value[:,(0,2)],pts2.value[:,(0,2)])


    
    def _get_roc(self,parametric_pts):
        '''gets the radius of curvature in the plane of a parametric 
        direction at specified parametric pts '''
        v_prime = self.geometry.evaluate(parametric_pts, parametric_derivative_orders=(0,1), non_csdl=True)
        v_double_prime = self.geometry.evaluate(parametric_pts, parametric_derivative_orders=(0,2), non_csdl=True)
        #TOTAL HACK: if either parametric derivative value is 0, set to some small number to prevent divide by zero
        
        roc = ( ((v_prime[:,0]**2 + v_prime[:,2]**2)**(3/2)) /
                (v_prime[:,0]*v_double_prime[:,2] - v_prime[:,2]*v_double_prime[:,0] ) )
        return roc
         
                
    def _get_pts_and_offset_pts(self,parametric_pts,layer=0,return_indices=False):
        ''' 
        Offset a set of points on a curve by the thickness assigned in the geometry
        parametric_pts: list of tuples of (surf_id, array(parametric_x,parametric_y))
        '''
        #TODO: make sure that the correct axial coordinate is added back to the final offset
        #       (not just zeros)
        #TODO: adjust to accomodate an arbitrary rotor axis via dot product with a specified axis
        #       (not just always assuming the y-axis, like at the moment is done)
        num_points = len(parametric_pts)
        pts = self.geometry.evaluate(parametric_pts, plot=False)
        normals = self.geometry.evaluate_normals(parametric_pts,plot=False)    
        if layer=='all':       
            thicknesses = self.quantities.material_properties.evaluate_thickness(parametric_pts)
        else:
            thicknesses = self._get_thicknesses(parametric_pts,layer)

        #perfom adjustment of in-plane normals to increase inplane thicknesses as req'd
        offset_thicknesses = thicknesses / csdl.sqrt(1- (normals[:,1])**2)

        #get inplane normals
        normals_inplane = (normals@(np.array([[1,0,0],[0,0,1]])).T)

        #offset inplane points by the inplane offset thickness
        #broadcasting vector to matrix not working well in csdl, so have to use expand
        offsets_inplane = normals_inplane* csdl.expand(offset_thicknesses,
                                                            normals_inplane.shape,
                                                            'i->ij')
        
        #add back the axial coordinate
        offsets = np.concatenate([offsets_inplane[:,0].value.reshape((num_points,1)),
                                        np.zeros((num_points,1)),
                                        offsets_inplane[:,1].value.reshape((num_points,1))],
                                        axis=1)
        #get radius of curvature
        roc = self._get_roc(parametric_pts)

        #discasrd offset points that would produce self-intersections
        #add some margin to the thicknesses to prevent missed invalid points due to errors in roc computation
        margin=2
        valid_offset_indices = (np.abs(roc) >= 2*thicknesses.value).nonzero()[0]
        offset_pts=( pts + offsets)[list(valid_offset_indices),:]
        # offset_pts = ( pts + offsets)

        if return_indices:
            return pts,offset_pts,valid_offset_indices
        else:
            return pts,offset_pts
        
    def _fit_xs_surface(self,pts,offset_pts,xs_surf_name,num_parametric):
        num_through_thickness = 2 #this is the number of ctl pts to fit the surface
        # lfs.BSplineSpace(2, (1, 1), (num_spanwise, 2))
        inplane_space = lfs.BSplineSpace(2,(3,1),(num_parametric//4,num_through_thickness))
        # stacked_pts = csdl.Variable(value=np.concatenate([top_pts[:,[0,2]].value,top_pts_offset.value]))
        stacked_pts = np.concatenate([pts.value,offset_pts.value])
        v_coords_pts = np.linspace(0,1,pts.shape[0])
        v_coords_pts_offset = np.linspace(0,1,offset_pts.shape[0])
        pts_parametric = np.concatenate([v_coords_pts.reshape((v_coords_pts.shape[0],1)),np.zeros((v_coords_pts.shape[0],1))],axis=1)
        pts_offset_parametric = np.concatenate([v_coords_pts_offset.reshape((v_coords_pts_offset.shape[0],1)),np.ones((v_coords_pts_offset.shape[0],1))],axis=1)
        parametric_coords = np.concatenate([pts_parametric,pts_offset_parametric],axis=0)
        
        surf_xs_coeffs = inplane_space.fit(values=stacked_pts,parametric_coordinates=parametric_coords)
        surf_xs = lfs.Function(inplane_space,surf_xs_coeffs)
        
        self._add_geometry(surf_index=self.xs_surf_index,
                        function=surf_xs,
                        name=xs_surf_name)
        self.xs_surf_index+=1

        xs_surf_geometry = self.create_subgeometry(search_names=xs_surf_name)

        return xs_surf_geometry
    

    def _mesh_curve_and_offset(self,pts,offset_pts,name='blade_xs',plot=False,meshsize=1,num_boundary_comp=1):
        ''' returns a mesh of a thickened curve in the plane'''
        gmsh.initialize()
        #initaite list of points
        pts_list = []
        pts_offset_list = []
        #need to loop through individually,as pts and offset_pts may be different lengths
        for pt in pts.value:
            pts_list.append(gmsh.model.occ.add_point(pt[0],pt[1],pt[2]))
        for offset_pt in offset_pts.value:
            pts_offset_list.append(gmsh.model.occ.add_point(offset_pt[0],offset_pt[1],offset_pt[2]))
        #ensure that the pts and offset points make a fully enclosed loop
        if num_boundary_comp==1:
            spline = gmsh.model.occ.add_spline(pts_list)
            spline_offset = gmsh.model.occ.add_spline(pts_offset_list)
            line1 = gmsh.model.occ.add_line(pts_offset_list[0],pts_list[0])
            line2 = gmsh.model.occ.add_line(pts_list[-1],pts_offset_list[-1],)
            
            CL1 = gmsh.model.occ.add_curve_loop([line1,spline,line2,-spline_offset])
        elif num_boundary_comp == 2:
            #ensure that the pts and offset points make a fully enclosed loop
            pts_list[-1] = pts_list[0]
            pts_offset_list[-1] = pts_offset_list[0]
            spline = gmsh.model.occ.add_spline(pts_list)
            spline_offset = gmsh.model.occ.add_spline(pts_offset_list)
            line1 = gmsh.model.occ.add_line(pts_list[0],pts_offset_list[0])
            
            CL1 = gmsh.model.occ.add_curve_loop([-line1,spline,line1,spline_offset])
        else:
            raise ValueError('Number of boundary components greater than 2 not supported!')
        
        surf = gmsh.model.occ.add_plane_surface([CL1])
        gmsh.model.add_physical_group(2,[surf],name="surface")

        #remove the pts so the import back into caddee isn't filled with thousands of pts
        for pt in pts_list + pts_offset_list:
            gmsh.model.occ.remove([(0, pt)])
        
        gmsh.model.occ.synchronize()
        
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize*.8)
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize*1.2)
        gmsh.option.set_number("Mesh.SaveAll", 2)
        gmsh.model.mesh.generate(2)
        if plot:
            gmsh.fltk.run()

        filename = "stored_files/"+ name + '.msh'
        gmsh.write(filename)
        filename2 = "stored_files/"+ name + '.vtk'
        gmsh.write(filename2)
        gmsh.finalize()

        return  filename
    
    def _mesh_curve_loop(self,curves,name='blade_xs',plot=False,meshsize=1,):
        ''' returns a mesh of a thickened curve in the plane'''
        gmsh.initialize()
        #initiate list of points
        pts_list = []
        splines = []

        # as points are added, make sure that spline endpoints match
        # use gmsh.model.get_value(dim,pt_tag,[])
        for j,pts in enumerate(curves):
            curve_pts_list = []

            for k,pt in enumerate(pts.value):
                #set the starting point tag of all splines after the first one to the previous splines endpoint
                if k == 0 and j!=0:
                    gmsh.model.occ.synchronize()
                    prev_boundary_pts = gmsh.model.getBoundary([(1, splines[j-1])], oriented=True, recursive=False)
                    # pt_coords= gmsh.model.get_value(0,prev_boundary_pts[0][1],[])
                    pt_tag = prev_boundary_pts[1][1]              
                #close the curve by setting the last point tag on the last spline to the tag of the first point on the first spline
                elif k == pts.shape[0]-1 and j == len(curves)-1:
                    gmsh.model.occ.synchronize()
                    starting_spline_boundary_pts= gmsh.model.getBoundary([(1, splines[0])], oriented=True, recursive=False)
                    # pt_coords = gmsh.model.get_value(0,starting_spline_boundary_pts[0][0])
                    pt_tag = starting_spline_boundary_pts[0][1]
                #otherwise, add a new point
                else:
                    pt_tag = gmsh.model.occ.add_point(pt[0],pt[1],pt[2])
                curve_pts_list.append(pt_tag)
                pts_list.append(pt_tag)
            splines.append(gmsh.model.occ.add_spline(curve_pts_list))                  
        
        CL1 = gmsh.model.occ.add_curve_loop(splines)
        
        surf = gmsh.model.occ.add_plane_surface([CL1])
        gmsh.model.add_physical_group(2,[surf],name="surface")

        #remove the pts so the import back into caddee isn't filled with thousands of pts
        for pt in pts_list:
            gmsh.model.occ.remove([(0, pt)])
        
        gmsh.model.occ.synchronize()
        
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize*.8)
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize*1.2)
        gmsh.option.set_number("Mesh.SaveAll", 2)
        gmsh.model.mesh.generate(2)
        if plot:
            gmsh.fltk.run()

        filename = "stored_files/"+ name + '.msh'
        gmsh.write(filename)
        filename2 = "stored_files/"+ name + '.vtk'
        gmsh.write(filename2)
        gmsh.finalize()

        return  filename
    
    def roc(self,geom, point):
        u_prime = geom.evaluate(point, parametric_derivative_orders=(0,1), non_csdl=True)[:,0]
        u_double_prime = geom.evaluate(point, parametric_derivative_orders=(0,2), non_csdl=True)[:,0]
        return np.abs(1+np.abs(u_prime)**(1/2)/u_double_prime)

    def _get_intersection_idx(self,curve_pts,intersection_pt):
        ''' find where an additional point should be store to add to the b-spline curve'''
        insert_idx=np.argmin(np.linalg.norm(curve_pts - intersection_pt,axis=1))
        #check if the insert idx is at the start or the end of the list
        if insert_idx == 0:
            insert_idx #unchanged
        elif insert_idx == curve_pts.shape[0]-1:
            insert_idx-=1
        else:
            #check surrounding points to determine which side of closest point to insert spar point
            surrounding_indices = [insert_idx-1,insert_idx+1]
            surrounding_to_insert = np.linalg.norm(curve_pts[surrounding_indices,:] - curve_pts[insert_idx,:],axis=1)
            surrounding_to_spar = np.linalg.norm(curve_pts[surrounding_indices,:] - intersection_pt,axis=1)
            if np.argmin(surrounding_to_insert-surrounding_to_spar) == 0:
                insert_idx = insert_idx+1
            elif np.argmin(surrounding_to_insert-surrounding_to_spar) == 1:
                insert_idx=insert_idx
        
        return insert_idx
    
    def _get_thicknesses(self,parametric_pts,layer):
        surf_index = parametric_pts[0][0]
        material_stack = self.quantities.material_properties.get_material_stack(surf_index)
        thicknesses = material_stack[layer]['thickness'].evaluate(parametric_pts)
        return thicknesses

#TODO: move these utility functions to a different module

#UTILITY FUNCIONS:



def chain_curves(curve_tags):
    # Build a mapping from curve tag -> (start_point, end_point)
    endpoints = {}
    for tag in curve_tags:
        b = gmsh.model.getBoundary([(1, tag)], oriented=True, recursive=False)
        start, end = b[0][1], b[1][1]
        endpoints[tag] = (start, end)
    
    # Start with any curve
    remaining = set(curve_tags)
    chained = []

    current_tag = remaining.pop()
    current_start, current_end = endpoints[current_tag]
    chained.append(current_tag)

    while remaining:
        found = False
        for tag in list(remaining):
            start, end = endpoints[tag]
            if start == current_end:
                # Direct match, add as is
                chained.append(tag)
                current_end = end
                remaining.remove(tag)
                found = True
                break
            elif end == current_end:
                # Reverse match, flip orientation
                chained.append(-tag)
                current_end = start
                remaining.remove(tag)
                found = True
                break
        if not found:
            raise RuntimeError("Cannot chain curves: they don't form a closed loop.")

    # Final check: does the last endpoint match the first starting point?
    first_start, _ = endpoints[abs(chained[0])]
    if current_end != first_start:
        raise RuntimeError("Curves do not form a closed loop after chaining.")

    return chained

def get_extension_vector(T,n,axis=1):
    '''
    T: skin tangent vector
    n: spar normal direction vector
    '''
    x = csdl.inner(n,T)
    if axis==0:
        #TODO:update this for x
        e =(csdl.sqrt(1-x**2)/x) * n[[2,1,0]]*np.array([-1,0,1])
    elif axis==1:
        e=(csdl.sqrt(1-x**2)/x) * n[[2,1,0]]*np.array([-1,0,1])
    elif axis==2:
        #TODO:update this for z
        e=(np.sqrt(1-x**2)/x) * n[[2,1,0]]*np.array([-1,0,1])
    return e

def rotate_oblique(vec,axis=1):
    scale = 1/csdl.sqrt(1-vec[:,axis]**2)
    if axis==0:
        adjustment=csdl.vstack([np.zeros(scale.shape),scale,scale]).T() #zero out x axis
    elif axis==1:
        adjustment=csdl.vstack([scale,np.zeros(scale.shape),scale]).T() #zero out y axis
    elif axis==2:
        adjustment=csdl.vstack([scale,scale,np.zeros(scale.shape)]).T() #zero out z axis
    return vec*adjustment

#Set a consistent value for all spar surface curves:
def set_axial_value(vec,val):
    vec.set_value(np.vstack([vec[:,0].value,
                                np.ones_like(vec[:,1].value)*val,
                                vec[:,2].value]).T)
    return
            # top_surf_xs.project
            # in progress
            # stacked_pts = #top_pts and top_pts_offset (make a variable )
            # parametric_coords = #0s and v_coords stacked with 1s and v_coords
            # top_surf_xs = inplane_space.fit_function(stacked_pts,parametric_coords) 
            # 
            # top_skin_mesh= #msh from gmsh (done once in setup)
            # top_surf_parametric_mesh = top_surf_xs.project(top_skin_mesh) # (done once in setup)
            # top_surf_mesh = top_surf_xs.evaluate(top_surf_parametric_mesh) #this is a csdl mesh that changes with the thickness

            # print("top surface thicknesses:")
            # print(self.quantities.material_properties.evaluate_thickness(parametric_top).value)
            # print('top pts:')
            # print(top_pts.value)
            # print('top normals:')
            # print(top_normals.value)
            # print('offset thicknesses:')
            # print(top_offset_thicknesses.value)
            # # top_normals_inplane=top_normals[:,(0,2)]
            # # top_pts_offset=top_normals_inplane*thickness + top_pts[:,[0,2]]
            # print('top offset points')
            # print(top_pts_offset.value)
            
            # #this would work if there was not curvature along the beam axis
            # # top_pts_offset = top_pts_normals*thickness + top_pts            
            # #this projects the normals into the plane and renormalizes to a uniform thickness, but this doesn't correctly capture the actual thickness at this point
            # # top_normals_inplane_normalized = ( (top_normals@(np.array([[1,0,0],[0,0,1]])).T) 
            # #  * ( 1/np.linalg.norm((top_normals@(np.array([[1,0,0],[0,0,1]])).T),axis=1).reshape((4,1)) )  )
            # #this projects the normals, but does not maintain the same offset distance as the original surface
            # top_normals_inplane = (top_normals@(np.array([[1,0,0],[0,0,1]])).T)
            # top_pts_offset = top_normals_inplane*thickness + top_pts[:,[0,2]]      
            # # top_thickness = self.quantities.material_properties.add_material()
            # # top_thicknesses = self.quantities.material_properties.evaluate_thickness(parametric_top)
            # # bot_thicknesses = self.quantities.material_properties.evaluate_thickness(parametric_bot)
            # print('offset inplane thickness coords:')
            # print((top_normals_inplane*thickness).value)
            # print('offset inplane thicknesses:')
            # print(np.linalg.norm(((top_normals_inplane*thickness).value),axis=1))
            # # print(csdl.norm((top_normals_inplane*thickness),axes=(1)).value)
            # print("top pts offset inplane:")
            # print(top_pts_offset.value)

            # pts=[(top_pts,top_pts_offset),bot_pts]
            # if front_spar_geometry is not None:
            #     #store front spar pts
            #     parametric_front_spar = [(front_spar_index, np.array([u_coord, v_coord])) for v_coord in v_coords]
            #     front_spar_pts = self.geometry.evaluate(parametric_front_spar, plot=False)
            #     pts.append(front_spar_pts)

            #     #add the start and end points to the top and bottom surface to enforce intersection
            #     #front spar point added to top curve
            #     insert_idx=np.argmin(np.linalg.norm(top_pts.value - front_spar_pts.value[0,:],axis=1))
            #     #check if the insert idx is at the start or the end of the list
            #     if insert_idx == 0:
            #         continue
            #     elif insert_idx == top_pts.shape[0]-1:
            #         insert_idx-=1
            #     else:
            #         #check surrounding points to determine which side of closest point to insert spar point
            #         surrounding_indices = [insert_idx-1,insert_idx+1]
            #         surrounding_to_insert = np.linalg.norm(top_pts.value[surrounding_indices,:] - top_pts.value[insert_idx,:],axis=1)
            #         surrounding_to_spar = np.linalg.norm(top_pts.value[surrounding_indices,:] - front_spar_pts.value[0,:],axis=1)
            #         if np.argmin(surrounding_to_insert-surrounding_to_spar) == 0:
            #             insert_idx = insert_idx+1
            #         elif np.argmin(surrounding_to_insert-surrounding_to_spar) == 1:
            #             insert_idx=insert_idx

            #     insertion_pts[i,0] = insert_idx
               
            #     #front spar point added to bottom curve
            #     insert_idx=np.argmin(np.linalg.norm(bot_pts.value - front_spar_pts.value[-1,:],axis=1))
            #     # insert_start,insert_end = np.sort(sorted_indices[0:2])
            #     #check if the insert idx is at the start or the end of the list
            #     if insert_idx == 0:
            #         continue
            #     elif insert_idx == bot_pts.shape[0]-1:
            #         insert_idx-=1
            #     else:
            #         #check surrounding points to determine which side of closest point to insert spar point
            #         surrounding_indices = [insert_idx-1,insert_idx+1]
            #         surrounding_to_insert = np.linalg.norm(bot_pts.value[surrounding_indices,:] - bot_pts.value[insert_idx,:],axis=1)
            #         surrounding_to_spar = np.linalg.norm(bot_pts.value[surrounding_indices,:] - front_spar_pts.value[-1,:],axis=1)
            #         if np.argmin(surrounding_to_insert-surrounding_to_spar) == 0:
            #             insert_idx = insert_idx+1
            #         elif np.argmin(surrounding_to_insert-surrounding_to_spar) == 1:
            #             insert_idx=insert_idx

            #     insertion_pts[i,1] = insert_idx

            # if rear_spar_geometry is not None:
            #     #store spar
            #     parametric_rear_spar = [(rear_spar_index, np.array([u_coord, v_coord])) for v_coord in v_coords]
            #     rear_spar_pts = self.geometry.evaluate(parametric_rear_spar, plot=False)
            #     pts.append(rear_spar_pts)
            
            # xs.append(pts)

        # from OCC.Core.gp import gp_Pnt2d
        # from OCC.Core.TColgp import TColgp_Array1OfPnt2d
        # from OCC.Core.Geom2dAPI import Geom2dAPI_PointsToBSpline,Geom2dAPI_InterCurveCurve
        # from OCC.Core.Geom2d import Geom2d_OffsetCurve,Geom2d_TrimmedCurve
        # from OCC.Display.SimpleGui import init_display
        # from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,BRepBuilderAPI_MakeFace
        # from OCC.Extend.ShapeFactory import make_wire, make_edge2d,make_edge
        # from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffset
        # # from OCC.Core.GeomAbs import GeomAbs_JoinType
        # from OCC.Core.GeomAbs import GeomAbs_Arc,GeomAbs_C2
        # from OCC.Core.BRepOffset import BRepOffset_Skin

        # bspline_order = 3
        # display, start_display, add_menu, add_function_to_menu = init_display()
        # pt_num = 1

        # #TODO: Need to add some checks here for a successful offset operation and determine whether
        # #   there are multiple bspline curves (e.g. a check for ISCLOSED)
        # for i in [3]:
        #     (top_pts_csdl,top_pts_offset_csdl) = xs[i][0]
        #     top_pts = top_pts_csdl.value
        #     top_pts_offset=top_pts_offset_csdl.value
        #     bot_pts = xs[i][1].value

        #     top_pt_array = TColgp_Array1OfPnt2d(pt_num, pt_num+num_parametric-1)
        #     for top_pt in top_pts:
        #         top_pt_array.SetValue(pt_num, gp_Pnt2d(top_pt[0], top_pt[2]))
        #         pt_num +=1

        #     top_pt_offset_array = TColgp_Array1OfPnt2d(pt_num, pt_num+num_parametric-1)
        #     for top_pt_offset in top_pts_offset:
        #         top_pt_offset_array.SetValue(pt_num, gp_Pnt2d(top_pt_offset[0], top_pt_offset[1]))
        #         pt_num +=1
            
        #     top_spline = Geom2dAPI_PointsToBSpline(top_pt_array, bspline_order, bspline_order)
        #     top_spline_offset = Geom2dAPI_PointsToBSpline(top_pt_offset_array, bspline_order, bspline_order)
        #     # top_spline_edge = BRepBuilderAPI_MakeEdge(top_spline).Edge()
        #     # top_spline_wire = BRepBuilderAPI_MakeWire(top_spline_edge).Wire()
        #     # display.DisplayShape(top_spline_wire)
        #     offset_distance = 0.0005
        #     offset_upper = 0.0005
        #     offset_lower = 0.0005
        #     offset_spar = 0.005
        #     offset_spar_cap = 0.006
        #     offset_tol = 1e-10

        #     # top_spline_offset = Geom2d_OffsetCurve(top_spline.Curve(), offset_upper)

        #     bot_pt_array = TColgp_Array1OfPnt2d(pt_num, pt_num+num_parametric-1)

        #     for bot_pt in bot_pts:
        #         bot_pt_array.SetValue(pt_num, gp_Pnt2d(bot_pt[0], bot_pt[2]))
        #         pt_num+=1
        #     # bot_spline = Geom2dAPI_PointsToBSpline(bot_pt_array).Curve()
        #     bot_spline = Geom2dAPI_PointsToBSpline(bot_pt_array, bspline_order, bspline_order) 

        #     # bot_spline_offset = Geom2d_OffsetCurve(bot_spline.Curve(), offset_lower)
            
        #     wire = make_wire([make_edge2d(top_spline.Curve()), make_edge2d(bot_spline.Curve())])
        #     # offset_wire = BRepOffsetAPI_MakeOffset(wire,dist,GeomAbs_JoinType.GeomAbs_Intersection, False)
        #     # offset = BRepOffsetAPI_MakeOffset()
        #     # offset.Init(GeomAbs_Arc)
        #     # offset.AddWire(wire)
        #     # offset.Perform(dist)
        #     # offset_wire = offset.Shape()

        #     #BSPLINE OFFSET CONSTRUCTION HERE....
        #     from OCC.Core.BRepCheck import BRepCheck_Analyzer
        #     def check_geometry(shape):
        #         analyzer = BRepCheck_Analyzer(shape)
        #         return analyzer.IsValid()
            
        #     print("OML:")
        #     print(wire.Closed())

        #     offset_builder = BRepOffsetAPI_MakeOffset(wire, BRepOffset_Skin)
        #     offset_builder.Perform(-offset_distance,offset_tol)

        #     #check offsetbuilder finished successfully
        #     print('offset success?:')
        #     print(offset_builder.IsDone())

        #     print('OML wire validity:')
        #     print(check_geometry(wire))

        #     offset_wire = offset_builder.Shape()
            
        #     print('offset closed?:')
        #     print(offset_wire.Closed() )     

        #     from OCC.Core.BRep import BRep_Tool
        #     from OCC.Core.TopExp import TopExp_Explorer
        #     from OCC.Core.TopAbs import TopAbs_EDGE
        #     from OCC.Core.TopoDS import topods
        #     from OCC.Core.TColgp import TColgp_Array1OfPnt
        #     from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
        #     from OCC.Display.SimpleGui import init_display
        #     # from OCC.Core.GeomAPI import GeomAPI_Interpolate

        #     # Function to extract points from an edge
        #     def extract_points_from_edge(edge, num_samples=100):
        #         curve_handle, first, last = BRep_Tool.Curve(edge)
        #         if curve_handle is None:
        #             return []
                
        #         points = []
        #         step = (last - first) / num_samples
        #         for i in range(num_samples + 1):
        #             pnt = curve_handle.Value(first + i * step)
        #             points.append(pnt)
        #         return points

        #     # Function to convert wire to a list of points
        #     def wire_to_points(wire, num_samples=100):
        #         points = []
        #         explorer = TopExp_Explorer(wire, TopAbs_EDGE)
        #         num_edges = 0
        #         while explorer.More():
        #             edge = topods.Edge(explorer.Current())
        #             edge_points = extract_points_from_edge(edge, num_samples)
        #             points.extend(edge_points)
        #             explorer.Next()
        #             num_edges+=1
        #         print('num_edges:')
        #         print(num_edges)
        #         return points

        #     # Extract points from the offset wire
        #     points = wire_to_points(offset_wire,100)

        #     # Create TColgp_Array1OfPnt from points
        #     num_points = len(points)
        #     # array_of_points = TColgp_HArray1OfPnt(1, ndum_points)
        #     array_of_points = TColgp_Array1OfPnt(1, num_points)
        #     for j, pnt in enumerate(points):
        #         array_of_points.SetValue(j + 1, pnt)

        #     # Create a B-spline curve from the points
        #     # interpolator = GeomAPI_PointsToBSpline(array_of_points,3,8)
        #     interpolator = GeomAPI_PointsToBSpline(array_of_points,3,3,GeomAbs_C2,1e-6)
        #     if interpolator.IsDone():
        #         bspline_curve = interpolator.Curve()

        #         # Display the B-spline curve
        #         # display.DisplayShape(bspline_curve, update=True, color="black")
        #     else:
        #         raise RuntimeError("Failed to create B-spline curve from points")
        #     print('bspline curve offset closed?:')
        #     print(bspline_curve.IsClosed())

        #     bspline_wire = make_wire([make_edge(bspline_curve)])
        #     #..... TO HERE
            
        #     # intersector = Geom2dAPI_InterCurveCurve(top_spline_offset,bot_spline_offset)
        #     # intersections = []
        #     # if intersector.NbPoints() > 0:
        #     #     for j in range(1, intersector.NbPoints() + 1):
        #     #         intersections.append(intersector.Point(j))
        #     #         # Geom2d_TrimmedCurve(top_spline_offset,)
        #     # print(f'number of intersections {j}')

        #     # intersection = intersections[0]
        #     # param1_curve1, param2_curve1 = Geom2dAPI_InterCurveCurve.Parameters(top_spline_offset, intersection)
        #     # param1_curve2, param2_curve2 = Geom2dAPI_InterCurveCurve.Parameters(bot_spline_offset, intersection)
            
        #     # trimmed_curve1 = Geom2d_TrimmedCurve(top_spline_offset, top_spline_offset.FirstParameter(), param1_curve1)
        #     # trimmed_curve2 = Geom2d_TrimmedCurve(bot_spline_offset, bot_spline_offset.FirstParameter(), param1_curve2)

        #     display.DisplayShape(top_spline.Curve(),color='RED')
        #     display.DisplayShape(bot_spline.Curve(),color="YELLOW")
        #     display.DisplayShape(top_spline_offset.Curve(), color="BLACK")
        #     # display.DisplayShape(bot_spline_offset, color="Yellow")
        #     # display.DisplayShape(trimmed_curve1, color="Black")
        #     # display.DisplayShape(trimmed_curve2, color="Black")
        #     # display.DisplayShape(offset_wire,color="RED")
        #     # display.DisplayShape(bspline_wire,color='RED')
        
            
        #     if front_spar_geometry is not None and rear_spar_geometry is not None:
        #         front_spar_pts = xs[i][2].value
        #         rear_spar_pts = xs[i][3].value
                
        #         #front spar
        #         front_spar_pt_array = TColgp_Array1OfPnt2d(pt_num, pt_num+num_parametric-1)
        #         for front_spar_pt in front_spar_pts:
        #             front_spar_pt_array.SetValue(pt_num, gp_Pnt2d(front_spar_pt[0], front_spar_pt[2]))
        #             pt_num +=1
        #         front_spar_spline = Geom2dAPI_PointsToBSpline(front_spar_pt_array,bspline_order, bspline_order)
        #         display.DisplayShape(front_spar_spline.Curve(),color="BLUE")
        #         front_spar_offset=Geom2d_OffsetCurve(front_spar_spline.Curve(), -offset_spar)
        #         display.DisplayShape(front_spar_offset,color="GREEN")

        #         #rear spar
        #         rear_spar_pt_array = TColgp_Array1OfPnt2d(pt_num, pt_num+num_parametric-1)
        #         for rear_spar_pt in rear_spar_pts:
        #             rear_spar_pt_array.SetValue(pt_num, gp_Pnt2d(rear_spar_pt[0], rear_spar_pt[2]))
        #             pt_num +=1
        #         rear_spar_spline = Geom2dAPI_PointsToBSpline(rear_spar_pt_array,bspline_order, bspline_order)
        #         display.DisplayShape(rear_spar_spline.Curve(),color="BLUE")
        #         rear_spar_offset=Geom2d_OffsetCurve(rear_spar_spline.Curve(), -offset_spar)
        #         display.DisplayShape(rear_spar_offset,color="GREEN")
            
        #     display.FitAll()
        #     # display.DisplayShape(bot_spline_offset, color="BLUE")
            
        #     start_display()

        #     from OCC.Core.BRep import BRep_Builder
        #     from OCC.Core.TopoDS import TopoDS_Compound
        #     from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        #     # from OCC.Core.TopExp import TopExp_Explorer
        #     # from OCC.Core.TopAbs import TopAbs_EDGE
        #     # from OCC.Core.TopoDS import topods_Edge

        #     # Create a compound to hold multiple wires
        #     compound = TopoDS_Compound()
        #     builder = BRep_Builder()
        #     builder.MakeCompound(compound)

        #     # Add wires to the compound
        #     builder.Add(compound, wire)
        #     # builder.Add(compound, rear_spar_offset)
        #     # builder.Add(compound, offset_wire)
        #     # builder.Add(compound, combined_wire)
        #     # builder.Add(compound, bspline_wire)
            
        #     # Initialize the STEP writer
        #     step_writer = STEPControl_Writer()

        #     # Add the wire to the STEP writer
        #     step_writer.Transfer(compound, STEPControl_AsIs)

        #     # Write the file
        #     step_writer.Write("OML.stp")

        #     compound1 = TopoDS_Compound()
        #     builder1 = BRep_Builder()
        #     builder1.MakeCompound(compound1)

        #     # Add wires to the compound
        #     builder1.Add(compound1, bspline_wire)
        #     # builder.Add(compound, rear_spar_offset)
        #     # builder1.Add(compound1, offset_wire)
        #     # builder.Add(compound, combined_wire)
        #     # builder.Add(compound, bspline_wire)
            
        #     # Initialize the STEP writer
        #     step_writer1 = STEPControl_Writer()

        #     # Add the wire to the STEP writer
        #     step_writer1.Transfer(compound1, STEPControl_AsIs)

        #     # Write the file
        #     step_writer1.Write("OML_offset.stp")


        # gmsh.initialize()

        # # gmsh.open("output.stp")
        # OML = gmsh.model.occ.import_shapes("OML.stp")
        # OML_tags = [item[1] for item in OML]
        # OML_offset=gmsh.model.occ.import_shapes("OML_offset.stp")
        # OML_offset_tags = [item[1] for item in OML_offset]
        # CL1 = gmsh.model.occ.add_curve_loop(OML_tags)
        # CL2 = gmsh.model.occ.add_curve_loop(OML_offset_tags)
        # gmsh.model.occ.add_plane_surface([CL1,CL2])
        # gmsh.model.occ.synchronize()
        # print(gmsh.model.get_entities(1))
        # gmsh.option.setNumber("Mesh.MeshSizeMin", offset_distance/2)
        # gmsh.option.setNumber("Mesh.MeshSizeMax", offset_distance/2)
        # gmsh.model.mesh.generate(2)
        # gmsh.fltk.run()

        # for i,pts in enumerate(xs):
        #     top_pts = pts[0]
        #     bot_pts = pts[1]
        #     lc =0.001
        #     top_pts_list=[]
        #     bot_pts_list=[]

        #     #add upper and lower surface points
        #     for top_pt in top_pts.value:
        #         top_pts_list.append(gmsh.model.occ.add_point(top_pt[0],top_pt[1],top_pt[2]))
        #     for bot_pt in bot_pts.value:
        #         if np.equal(bot_pt,top_pts.value[0,:]).all():
        #             bot_pts_list.append(top_pts_list[0])
        #         elif np.equal(bot_pt,top_pts.value[-1,:]).all():
        #             bot_pts_list.append(top_pts_list[-1])
        #         else:
        #             bot_pts_list.append(gmsh.model.occ.add_point(bot_pt[0],bot_pt[1],bot_pt[2]))
            
        #     #check that xc_pts1 and 2 have the same start and end points
        #     # if not np.equal(top_pts[0,:].value,bot_pts[-1,:].value).all():
        #     if  top_pts_list[0] != bot_pts_list[-1]:
        #         bot_pts_list.append(top_pts_list[0])
        #     # if not np.equal(top_pts[-1,:].value,bot_pts[0,:].value).all():
        #     if  top_pts_list[-1] != bot_pts_list[0]:
        #         bot_pts_list.insert(0,top_pts_list[-1])
        #         if rear_spar_geometry and front_spar_geometry is not None:
        #             insertion_pts[i,1]+=1
                       
        #     # add curve loop
        #     # CL1 = gmsh.model.occ.add_curve_loop([spline1,spline2])

        #     # try to offset, but if offset returns nothing, make a solid section
        #     # CL2_tags_raw = gmsh.model.occ.offset_curve(CL1,.005)

        #     # gmsh.model.geo.add_plane_surface([spline])
        #     # gmsh.model.occ.add_plane_surface([CL1])
        #     # gmsh.model.occ.add_plane_surface([CL2])

        #     # #if the offset curve process fails, create a solid section
        #     # try:
        #     #     #need to get proper list of curves by removing dim part of each tuple
        #     #     CL2_tags = [curve[1] for curve in CL2_tags_raw ]
        #     #     CL2 = gmsh.model.occ.add_curve_loop(CL2_tags)
        #     #     gmsh.model.occ.add_plane_surface([CL1,CL2])
        #     # except:
        #     #     gmsh.model.occ.add_plane_surface([CL1])
            
        #     #if there are spars, add their geometry to the occ represenation
        #     if rear_spar_geometry and front_spar_geometry is not None:
        #         front_spar_pts = pts[2]
        #         front_spar_pts_list=[]
        #         for pt3 in front_spar_pts.value:
        #             front_spar_pts_list.append(gmsh.model.occ.add_point(pt3[0],pt3[1],pt3[2]))
        #         front_spar_spline = gmsh.model.occ.add_spline(front_spar_pts_list)
                
        #         # top_front_curve_pts = top_pts_list[0:insertion_pts[i,0]]
        #         # oml_points = top_pts_list + bot_pts_list[1:]
        #         top_pts_list.insert(insertion_pts[i,0],front_spar_pts_list[0])
        #         bot_pts_list.insert(insertion_pts[i,1],front_spar_pts_list[-1])
        #         top_spline = gmsh.model.occ.add_spline(top_pts_list)
        #         bot_spline = gmsh.model.occ.add_spline(bot_pts_list)

        #         # output = gmsh.model.geo.split_curve(top_spline,[0,front_spar_pts_list[0]])
                
        #         # print(output)
        #         # output = gmsh.model.occ.fragment([(1,top_spline)],[(1,front_spar_spline)],removeObject=False,removeTool=False)
        #         # output = gmsh.model.occ.fragment([(1,bot_spline)],[(1,front_spar_spline)],removeObject=False,removeTool=False)
        #         # output = gmsh.model.occ.cut([(1,top_spline)],[(0,front_spar_pts_list[0])],removeObject=True,removeTool=False)
        #         # output2 = gmsh.model.occ.cut([(1,bot_spline)],[(0,front_spar_pts_list[-1])],removeObject=True,removeTool=False)
        #         # output = gmsh.model.occ.cut([(1,bot_spline)],[(1,front_spar_spline)],removeObject=True,removeTool=False)

        #         # CL1 = gmsh.model.occ.add_curve_loop([front_spar_spline,5,6])
        #         # CL2 = gmsh.model.occ.add_curve_loop([front_spar_spline,4,7])
        #         CL3 = gmsh.model.occ.add_curve_loop([top_spline,bot_spline])
        #         gmsh.model.occ.offset_curve(CL3,-0.00001)
        #         # top_spar_curve_pts =
        #         # top_rear_curve_pts = 

        #         # gmsh.model.occ.add_plane_surface([CL1])

        #         # print(CL1)
        #         # print(CL2_tags_raw)
        #         # output = gmsh.model.occ.intersect(CL2_tags_raw,[(1,spline3)],False,False)
        #         # output = gmsh.model.occ.fragment([(1,spline1),(1,spline2)],[(1,spline3)])
        #         # output = gmsh.model.occ.cut([(1,spline1)],[(1,spline3)],removeObject=False,removeTool=False)
        #         # output2 = gmsh.model.occ.cut([(1,spline2)],[(1,spline3)],removeObject=False,removeTool=False)
        #         # print(output)
        #         # print(output2)
        #         # curve_tags = [curve[1] for curve in output[0]]
        #         # curve_tags.append(spline3)
        #         # FS_CL1 = gmsh.model.occ.add_curve_loop(curve_tags)
        #         # CL2_tags_raw = gmsh.model.occ.offset_curve(FS_CL1,-.005)
                
        #         rear_spar_pts = pts[3]
        #         pts4=[]
        #         for pt4 in rear_spar_pts.value:
        #             pts4.append(gmsh.model.occ.add_point(pt4[0],pt4[1],pt4[2]))
        #         rear_spar_spline = gmsh.model.occ.add_spline(pts4)



        #     else:
        #         #add the upper and lower skins splines
        #         top_spline = gmsh.model.occ.add_spline(top_pts_list)
        #         bot_spline = gmsh.model.occ.add_spline(bot_pts_list)

        #     # output = gmsh.model.occ.cut([(1,spline1)],[(1,spline3)],removeObject=True,removeTool=False)
        #     # output2 = gmsh.model.occ.cut([(1,spline2)],[(1,spline3)],removeObject=True,removeTool=False)
        #     # print(output)
        #     # print(output2)
        #     # print('pt lists')
        #     # print(top_pts_list)
        #     # print(bot_pts_list)

        #     # #this seems to work
        #     # CL1 = gmsh.model.occ.add_curve_loop([spline1,spline2])
        #     # gmsh.model.occ.add_plane_surface([CL1])
        #     # CL2_tags_raw = gmsh.model.occ.offset_curve(CL1,-.005)
            
        #     # output = gmsh.model.occ.intersect([(1,spline1),(1,spline2)],[(1,spline3)],removeObject=False,removeTool=False)
        #     # output = gmsh.model.occ.fragment([(1,spline1),(1,spline2)],[(1,spline3)])
        #     # print(output)

        #     gmsh.model.geo.synchronize()

        #     gmsh.model.occ.synchronize()

        #     gmsh.fltk.run()

        #     gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)
        #     gmsh.option.setNumber('Mesh.MeshSizeMax', 0.005)

        #     gmsh.model.occ.synchronize()

        #     gmsh.model.mesh.generate(2)

        #     gmsh.fltk.run()

        # gmsh.finalize()

        # return

                    # num_through_thickness = 2 #this is the number of ctl pts to fit the surface
            # # lfs.BSplineSpace(2, (1, 1), (num_spanwise, 2))
            # inplane_space = lfs.BSplineSpace(2,(3,1),(num_parametric//4,num_through_thickness))
            # # stacked_pts = csdl.Variable(value=np.concatenate([top_pts[:,[0,2]].value,top_pts_offset.value]))
            # stacked_pts = np.concatenate([top_pts.value,top_pts_offset.value])
            # # stacked_pts = csdl.Variable(value=stacked_pts)

            # # print(stacked_pts.value)
            # # top_pts_parametric = np.concatenate([np.zeros((v_coords.shape[0],1)),v_coords.reshape((v_coords.shape[0],1))],axis=1)
            # # top_pts_offset_parametric = np.concatenate([np.ones((v_coords.shape[0],1)),v_coords.reshape((v_coords.shape[0],1))],axis=1)
            # top_pts_parametric = np.concatenate([v_coords.reshape((v_coords.shape[0],1)),np.zeros((v_coords.shape[0],1))],axis=1)
            # top_pts_offset_parametric = np.concatenate([v_coords.reshape((v_coords.shape[0],1)),np.ones((v_coords.shape[0],1))],axis=1)
            # parametric_coords = np.concatenate([top_pts_parametric,top_pts_offset_parametric],axis=0)

            # # parametric_coords = csdl.Variable(value=parametric_coords)
            # # top_surf_xs = inplane_space.fit(values=stacked_pts,parametric_coordinates=parametric_coords)
          
            # # top_surf_xs = inplane_space.fit_function(values=stacked_pts,parametric_coordinates=parametric_coords) 
            # top_surf_xs_coeffs = inplane_space.fit(values=stacked_pts,parametric_coordinates=parametric_coords)
            # top_surf_xs = lfs.Function(inplane_space,top_surf_xs_coeffs)
            # top_surf_xs_name='top skin mesh'
            
            # self._add_geometry(surf_index=surf_index,
            #                     function=top_surf_xs,
            #                     name=top_surf_xs_name)
            # surf_index+=1


            # top_surf_xs_geometry = self.create_subgeometry(search_names=top_surf_xs_name)

            # fit_xs_surface(top_pts,top_pts_offset,surf_name=top_surf_xs_name,surf_index=surf_index)

            # top_xs_surf_geometry.plot(opacity=0.5)
            # bot_xs_surf_geometry.plot(opacity=0.5)
            # self.geometry.plot(opacity=0.5)

                    # if front_spar_geometry is not None and rear_spar_geometry is not None:
        #     insertion_pts = np.empty((u_coords.shape[0],4),dtype=int)
        
        # skin_thickness = 0.05

            # bot_pts = self.geometry.evaluate(parametric_bot, plot=False)
            # bot_normals = self.geometry.evaluate_normals(parametric_bot,plot=False)
            
            
            # # thickness.set_as_design_variable(upper=0.05, lower=0.0001, scaler=1e3)
            # function = lfs.Function(lfs.ConstantSpace(2), thickness)
            # functions = {top_index: function, bot_index: function}
            # thickness_fs = lfs.FunctionSet(functions)
            # # E = csdl.Variable(value=69E9, name='E')
            # # G = csdl.Variable(value=26E9, name='G')
            # # density = csdl.Variable(value=2700, name='density')
            # # nu = csdl.Variable(value=0.33, name='nu')
            # # aluminum = cd.materials.IsotropicMaterial(name='aluminum', density=density, E=E, nu=nu, G=G)
            # material = self.quantities.material_properties.material

            # self.quantities.material_properties.set_material(material, thickness_fs)
            
            # top_thicknesses = self.quantities.material_properties.evaluate_thickness(parametric_top)
            
            # top_offset_thicknesses = top_thicknesses / csdl.sqrt(1- (top_normals[:,1])**2)
            # # top_offset_thicknesses = thickness / csdl.sqrt(1- (top_normals[:,1])**2)
            # top_normals_inplane = (top_normals@(np.array([[1,0,0],[0,0,1]])).T)
            # #broadcasting vector to matrix not working well in csdl, so have to use expand
            # top_pts_offset_inplane = top_normals_inplane* csdl.expand(top_offset_thicknesses,
            #                                                     top_normals_inplane.shape,
            #                                                     'i->ij')
            # top_pts_offsets = np.concatenate([top_pts_offset_inplane[:,0].value.reshape((num_parametric,1)),
            #                                   np.zeros((num_parametric,1)),
            #                                   top_pts_offset_inplane[:,1].value.reshape((num_parametric,1))],
            #                                   axis=1)
            # top_pts_offset= ( top_pts +
            #                  top_pts_offsets)
            

            #ROC STUFF:
            # u_prime = self.geometry.evaluate(parametric_bot, parametric_derivative_orders=(0,1), non_csdl=True)
            # u_double_prime = self.geometry.evaluate(parametric_bot, parametric_derivative_orders=(0,2), non_csdl=True)
            
            # u_prime = ( self.geometry.evaluate(parametric_bot, parametric_derivative_orders=(0,1), non_csdl=True) +
            #             self.geometry.evaluate(parametric_bot, parametric_derivative_orders=(1,0), non_csdl=True) )
            # u_double_prime = ( self.geometry.evaluate(parametric_bot, parametric_derivative_orders=(0,2), non_csdl=True) +
            #                    self.geometry.evaluate(parametric_bot, parametric_derivative_orders=(2,0), non_csdl=True) )
            # roc = ((1+u_prime[:,0]**2)**(3/2))/u_double_prime[:,0]
            # print('min bot roc:')
            # print(np.min(np.abs(roc)))
            # # print(np.min(np.abs(((1+u_prime[:,1]**2)**(3/2))/u_double_prime[:,1])))
            # print(np.min(np.abs(((1+u_prime[:,2]**2)**(3/2))/u_double_prime[:,2])))
            # roc_bot = ( ((u_prime[:,0]**2 + u_prime[:,2]**2)**(3/2)) /
            #        (u_prime[:,0]*u_double_prime[:,2] - u_prime[:,2]*u_double_prime[:,0] ) )