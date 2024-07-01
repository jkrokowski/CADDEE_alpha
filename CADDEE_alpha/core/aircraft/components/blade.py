import lsdo_function_spaces as lfs
from lsdo_function_spaces import FunctionSet
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
                # print("time for copmuting wetted area", t2-t1)

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
    
    def construct_cross_section(
            self,
            geometry:lg.Geometry,
            top_geometry:lg.Geometry,
            bottom_geometry:lg.Geometry,
            spar_locations:np.ndarray=None,
            spar_function_space:lfs.FunctionSpace=None,
            surf_index:int=1000,
            offset:np.ndarray=np.array([0., 0., .23]), 

        ):
        '''
        Construct the internal geometry of the rotor blade based on input 
        parameters and the upper and lower surfaces of the blade 
        '''
        csdl.check_parameter(spar_locations, "spar_locations", types=np.ndarray, allow_none=True)
        csdl.check_parameter(surf_index, "surf_index", types=int)

        # Check if spar and rib locations are between 0 and 1
        if spar_locations is not None:
            if not np.all((spar_locations > 0) & (spar_locations < 1)):
                raise ValueError("all spar locations must be between 0 and 1 (excluding the endpoints)")
        if spar_locations is None:
            spar_locations = np.array([0.25, 0.75])
        
        blade = self
        
        num_spars = spar_locations.shape[0]
        if num_spars == 1:
            raise Exception("Cannot have single spar. Provide at least two normalized spar locations.")
        
        num_spanwise = 10
        if spar_function_space is None:
            spar_function_space = lfs.BSplineSpace(2, (1, 1), (num_spanwise, 2))

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
        print("root tip points")
        print(root_tip_pts[:,0])
        #compute projection direction (perpendicular to the chordline)
        chord_direction_root = root_le-root_te
        chord_direction_tip = r_tip_le-r_tip_te
        spanwise_direction = np.array([0,1,0])
        proj_direction_root = np.cross(chord_direction_root,spanwise_direction)
        proj_direction_tip = np.cross(chord_direction_tip,spanwise_direction)
        print("projecting...")
        print(root_tip_pts[:,0])
        print(proj_direction_root)
        ribs_top_root = top_geometry.project(root_tip_pts[:,0], direction=proj_direction_root, grid_search_density_parameter=10, plot=True)
        ribs_bottom_root = bottom_geometry.project(root_tip_pts[:,0], direction=-proj_direction_root, grid_search_density_parameter=10, plot=True)
        ribs_top_tip = top_geometry.project(root_tip_pts[:,1], direction=proj_direction_tip, grid_search_density_parameter=10, plot=True)
        ribs_bottom_tip = bottom_geometry.project(root_tip_pts[:,1], direction=-proj_direction_tip, grid_search_density_parameter=10, plot=True)

        
        parametric_LE = np.vstack([np.linspace(0,1,num_spanwise),np.ones((num_spanwise))]).T
        parametric_TE = np.vstack([np.linspace(0,1,num_spanwise),np.zeros((num_spanwise))]).T

        physical_LE = blade.geometry.evaluate([(174,parametric_LE)],plot=True,non_csdl=True)
        physical_TE = blade.geometry.evaluate([(174,parametric_TE)],plot=True,non_csdl=True)
        
        chord_direction = physical_LE-physical_TE
        # print(proj_direction_root)
        # print(proj_direction_tip)
        proj_direction = np.cross(chord_direction,np.tile(spanwise_direction,(num_spanwise,1)))
        # print(proj_direction)

        fwd_spar_chord_loc = (1-spar_locations[0]) * physical_LE + spar_locations[0] * physical_TE
        rear_spar_chord_loc = (1-spar_locations[1]) * physical_LE + spar_locations[1] * physical_TE
        print("Fwd spar locations")
        print(fwd_spar_chord_loc)
        print("Rear spar locations")
        print(rear_spar_chord_loc)

        # get spar start/end points (root and tip)
        # root_tip_pts = np.zeros((num_spars, num_spanwise, 3))
        for i in range(num_spanwise):
            # root_tip_pts[i,0] = (1-spar_locations[i]) * root_le + spar_locations[i] * root_te
            # root_tip_pts[i,1] = (1-spar_locations[i]) * r_tip_le + spar_locations[i] * r_tip_te
            #project up and down to get all points
            print("projecting...")
            print(fwd_spar_chord_loc[i,:])
            print(proj_direction[i,:])
            ribs_top_root = top_geometry.project(fwd_spar_chord_loc[i,:], direction=proj_direction[i,:], grid_search_density_parameter=10, plot=True)
            # ribs_bottom_root = bottom_geometry.project(root_tip_pts[:,0], direction=-proj_direction_root, grid_search_density_parameter=10, plot=True)
            # ribs_top_tip = top_geometry.project(root_tip_pts[:,1], direction=proj_direction_tip, grid_search_density_parameter=10, plot=True)
            # ribs_bottom_tip = bottom_geometry.project(root_tip_pts[:,1], direction=-proj_direction_tip, grid_search_density_parameter=10, plot=True)

        #create spar surfaces
        # for i in num_spars:
        #     parametric_points = ribs_top_array[:,i*num_rib_pts].tolist() + ribs_bottom_array[:,i*num_rib_pts].tolist()
        #     u_coords = np.linspace(0, 1, num_ribs)
        #     fitting_coords = np.array([[u, 0] for u in u_coords] + [[u, 1] for u in u_coords])
        #     spar, right_spar = self._fit_surface(parametric_points, fitting_coords, spar_function_space, True, True)
        #     self._add_geometry(surf_index, spar, "Blade_spar_", i)
        #     surf_index = self._add_geometry(surf_index, spar, "Wing_l_spar_", i, geometry)
        

    def create_beam_xs_meshes(
            self,
            top_index,
            bottom_index,
            num_spanwise
        ):
        u_coords = np.linspace(0,1,num_spanwise)
        v_coords = np.linspace(0,1,10)
        xcs = []
        for u_coord in u_coords:
            parametric_coordinates = [(top_index, np.array([u_coord, v_coord])) for v_coord in v_coords]
            parametric_coordinates_2 = [(bottom_index, np.array([u_coord, v_coord])) for v_coord in v_coords]

            pts1 = self.geometry.evaluate(parametric_coordinates, plot=False)
            pts2 = self.geometry.evaluate(parametric_coordinates_2, plot=False)

            xcs.append([pts1,pts2])

        gmsh.initialize()
        for xc_pts1,xc_pts2 in xcs:
            
            lc =0.001
            pts1=[]
            pts2=[]
            for pt1,pt2 in zip(xc_pts1.value,xc_pts2.value):
                # pts.append(gmsh.model.geo.add_point(pt[0],pt[1],pt[2],lc))
                pts1.append(gmsh.model.occ.add_point(pt1[0],pt1[1],pt1[2]))
                pts2.append(gmsh.model.occ.add_point(pt2[0],pt2[1],pt2[2]))
            #check that xc_pts1 and 2 have the same start and end points
            # if, not add it to one of the lower surface
            if not np.equal(xc_pts1[0,:].value,xc_pts2[-1,:].value).all():
                pt = xc_pts1[0,:].value
                pts2.append(gmsh.model.occ.add_point(pt[0],pt[1],pt[2]))
            if not np.equal(xc_pts1[-1,:].value,xc_pts2[0,:].value).all():
                pt = xc_pts1[-1,:].value
                pts2.insert(0,gmsh.model.occ.add_point(pt[0],pt[1],pt[2]))

            # pts.append(pts[0])

            # spline = gmsh.model.geo.add_spline(pts)
            spline1 = gmsh.model.occ.add_spline(pts1)
            spline2 = gmsh.model.occ.add_spline(pts2)
            # bspline = gmsh.model.geo.add_compound_bspline([spline])

            # gmsh.model.geo.add_curve_loop([spline])
            CL1 = gmsh.model.occ.add_curve_loop([spline1,spline2])

            #try to offset, but if offset returns nothing, make a solid section
            CL2_tags_raw = gmsh.model.occ.offset_curve(CL1,-.005)

            # gmsh.model.geo.add_plane_surface([spline])
            # gmsh.model.occ.add_plane_surface([CL1])
            # gmsh.model.occ.add_plane_surface([CL2])

            #if the offset curve process fails, create a solid section
            try:
                #need to get proper list of curves by removing dim part of each tuple
                CL2_tags = [curve[1] for curve in CL2_tags_raw ]
                CL2 = gmsh.model.occ.add_curve_loop(CL2_tags)
                gmsh.model.occ.add_plane_surface([CL1,CL2])
            except:
                gmsh.model.occ.add_plane_surface([CL1])

        gmsh.option.setNumber('Mesh.MeshSizeMin', 0.001)
        gmsh.option.setNumber('Mesh.MeshSizeMax', 0.005)

        # gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(2)

        gmsh.fltk.run()

        gmsh.finalize()


        return
    
# class CrossSection(Component):
#     def __init__(self, **kwargs):
