#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

class Road:
    """
    This class provides a road-like environment that is described by a sequnce
    of straight-line and curved elements. The method for constructing a complete
    road ensure that the road is continuous and smooth. The class also includes
    function for computing the details of the road relative to a given query
    point, i.e., the details that a car may be able to measure while driving
    along the road.

    Quantities that fully specify each road element
    - c_i : the curvature of the road element (units: 1/m).
    - l_i : the length of the road element (units: m).

    From c_i and l_i, the following quantities are computed:
    - phi_i      : the angular span of curved road elements as theta_i = l_i * c_i, equal to NaN for a straight-line element (units: radians)
    - isStraight_i : A flag for whether the road element is straight, by checking |c_i| < epsilon (units: boolean)

    Additional quantities are derived from the above values as they are convenient for the
    functions that process the road:
    - start_point_i        : (x,y) coordinates of the start of the road element (units: m).
    - end_point_i          : (x,y) coordinates of the end   of the road element (units: m).
    - start_angle_i        : angle (relative to world frame x-axis) of the tangent to the road element at its start point (units: radians).
    - end_angle_i          : angle (relative to world frame x-axis) of the tangent to the road element at its end.  point (units: radians).
    - arc_center_i         : (x,y) coordinates of the center point of a curved road element (units: m).
    - l_total_at_end_i     : the total length of the road at the end point that the road element (units: m).
    - start_hyperplane_A_i : the unit normal vector that point in the opposite direction of start angle (units: m).
    - start_hyperplane_b_i : the offset value such that the hyperplane pass through the start point (units: m).
    - end_hyperplane_A_i   : the unit normal vector that point in the opposite direction of end angle (units: m).
    - end_hyperplane_b_i   : the offset value such that the hyperplane pass through the end point (units: m).
    """



    def __init__(self, epsilon_c=1/10000):
        """
        Initialization function for the "Road" class.

        Parameters
        ----------
            epsilon_c : float
                The threshold for the minimum allow value of curvature of a
                road element. Road element with curvature below this threshold
                are set to a curvature of zero, i.e., a straight line. This
                threshold is imposed to avoid nuermcial complication when
                computing quantities that depend on the inverse of the curvature
                (i.e., the radius).

        Returns
        -------
        Nothing
        """
        # Set the epsilon
        self.epsilon_c= epsilon_c

        # Initialize empty vectors for each property of the road element
        # > As private class variables
        self.__c = np.empty((0,),dtype=np.float32)
        self.__l = np.empty((0,),dtype=np.float32)
        self.__phi = np.empty((0,),dtype=np.float32)
        self.__isStraight = np.empty((0,),dtype=np.bool_)

        # Initialize empty vectors for the derived properties of the road elements
        # > As private class variables
        self.__start_points = np.empty((0,2),dtype=np.float32)
        self.__end_points = np.empty((0,2),dtype=np.float32)
        self.__start_angles = np.empty((0,),dtype=np.float32)
        self.__end_angles = np.empty((0,),dtype=np.float32)
        self.__arc_centers = np.empty((0,2),dtype=np.float32)
        self.__l_total_at_end = np.empty((0,),dtype=np.float32)
        self.__start_hyperplane_A = np.empty((0,2),dtype=np.float32)
        self.__start_hyperplane_b = np.empty((0,),dtype=np.float32)
        self.__end_hyperplane_A = np.empty((0,2),dtype=np.float32)
        self.__end_hyperplane_b = np.empty((0,),dtype=np.float32)


    def get_c(self): return np.copy(self.__c)
    def get_l(self): return np.copy(self.__l)
    def get_phi(self): return np.copy(self.__phi)
    def get_isStraight(self): return np.copy(self.__isStraight)
    def get_start_points(self): return np.copy(self.__start_points)
    def get_end_points(self): return np.copy(self.__end_points)
    def get_start_angles(self): return np.copy(self.__start_angles)
    def get_end_angles(self): return np.copy(self.__end_angles)
    def get_arc_centers(self): return np.copy(self.__arc_centers)
    def get_l_total_at_end(self): return np.copy(self.__l_total_at_end)
    def get_start_hyperplane_A(self): return np.copy(self.__start_hyperplane_A)
    def get_start_hyperplane_b(self): return np.copy(self.__start_hyperplane_b)
    def get_end_hyperplane_A(self): return np.copy(self.__end_hyperplane_A)
    def get_end_hyperplane_b(self): return np.copy(self.__end_hyperplane_b)

    def get_total_length(self):
        return self.__l_total_at_end[-1]

    def add_road_element_straight(self, length=100):
        """
        Appends a straight-line road element to the end of the current road.

        Parameters
        ----------
            length : float
                The length of the road element to be added (units: m).

        Returns
        -------
        Nothing
        """
        # Check that the length is greater than zero
        if (length <= 0):
            print("ERROR: Road elements must have a positive length. Hence, no road element added.")
            return

        # Append the details of this element
        self.__c   = np.append(self.__c  , 0.)
        self.__l   = np.append(self.__l  , length)
        self.__phi = np.append(self.__phi, np.nan)
        self.__isStraight = np.append(self.__isStraight, True)

        # Compute the derived properties for this element
        self.__add_derived_properties_of_road_element((self.__c.shape[0]-1))



    def add_road_element_curved_by_length(self, curvature=1/100, length=100):
        """
        Appends a circular-arc road element to the end of the current road, with
        length specified by distance along the arc that the road element traces.

        Parameters
        ----------
            curvature : float
                The curvature of the road element to be added, which is the
                inverse of the radius (units: 1/m).
            length : float
                The arc length of the circular road element to be added (units: m).

        Returns
        -------
        Nothing
        """
        # Check that the length is greater than zero
        if (length <= 0):
            print("ERROR: Road elements must have a positive length. Hence, no road element added.")
            return

        # Check is the curvature is greater than the minimum
        if (abs(curvature) < self.epsilon_c):
            print("WARNING: Curved road element must have cuurvature greater than " + str(self.epsilon_c) + ". Hence, increasing the curvature of this element to the minimum.")
            curvature = np.sign(curvature) * self.epsilon_c

        # Compute the angular space of this road element
        angular_span = length * abs(curvature)

        # Warn the user if the angular span is greater than 2pi
        if (angular_span >= 2*np.pi):
            print("WARNING: Curved road element specified with angular span of " + "{:.1f}".format(angular_span*(180/np.pi)) + ", which is great than 0.5 revolution, which the limits of this class. Adding the element anyway")

        # Append the details of this element
        self.__c   = np.append(self.__c  , curvature)
        self.__l   = np.append(self.__l  , length)
        self.__phi = np.append(self.__phi, angular_span)
        self.__isStraight = np.append(self.__isStraight, False)

        # Compute the derived properties for this element
        self.__add_derived_properties_of_road_element((self.__c.shape[0]-1))



    def add_road_element_curved_by_angle(self, curvature=1/100, angle_in_degrees=45):
        """
        Appends a circular-arc road element to the end of the current road, with
        length specified by the anglar span of the arc.

        Parameters
        ----------
            curvature : float
                The curvature of the road element to be added, which is the
                inverse of the radius (units: 1/m).
            angle_in_degrees : float
                Angular span of the circular arc from start to end of the road
                element (units: degrees).
                Note: arc length = angular span (in radians) * radius

        Returns
        -------
        Nothing
        """
        # Convert the angle to arc length and call the other function
        self.add_road_element_curved_by_length( curvature=curvature, length=(np.pi/180)*angle_in_degrees/abs(curvature))



    def __add_derived_properties_of_road_element(self, for_index=-1):
        """
        Computes the "derived properties" of a road element and append that
        values to the end of the class properties that store these values.

        Note: it is assumed (and required) that this function is called
        directly after appending each road element.

        Parameters
        ----------
            for_index : integer
                The index of the road element for which the derived properties
                are to be computed.

        Returns
        -------
        Nothing
        """
        # Check the  derived properties are the expected length

        if not(self.__start_points.shape[0] == for_index):
            print("ERROR: The length of the derived properties is out of sync with the length of the road elements added.")
            return

        # The start point is the end point of the previous element
        # > When the first element is forced to start at (0,0)
        if (for_index == 0):
            this_start_point = np.array([[0.,0.]],dtype=np.float32)
            this_start_angle = np.array([0.],dtype=np.float32)
        else:
            this_start_point = np.copy(self.__end_points[(for_index-1):for_index,:])
            this_start_angle = np.copy(self.__end_angles[(for_index-1):for_index])

        # Derived properties for a straight line
        if (self.__isStraight[for_index]):
            # Set the end angle equal to the start angle
            this_end_angle = np.copy(this_start_angle)
            # Set the arc center to be not necessary for a straight element
            this_arc_center = np.array([[np.nan, np.nan]], dtype=np.float32)
            # Compute the end point based on the distance on the straight element
            this_end_point = this_start_point + self.__l[for_index] * np.array([[np.cos(this_start_angle[0]),np.sin(this_start_angle[0])]], dtype=np.float32)
        else:
            # Compute the end angle based on the phi of the curved element
            this_end_angle = np.copy(this_start_angle) + np.sign(self.__c[for_index]) * self.__phi[for_index:(for_index+1)]
            # > Unwrap the angle to be in the range [-pi,pi]
            while (this_end_angle[0] < -np.pi):
                this_end_angle[0] = this_end_angle[0] + 2*np.pi
            while (this_end_angle[0] > np.pi):
                this_end_angle[0] = this_end_angle[0] - 2*np.pi

            # Compute the arc center by tracing the radius from the start point
            angle_start_to_center = this_start_angle[0] + np.sign(self.__c[for_index]) * 0.5 * np.pi
            this_arc_center = this_start_point + (1/abs(self.__c[for_index])) * np.array([[np.cos(angle_start_to_center),np.sin(angle_start_to_center)]], dtype=np.float32)

            # Compute the end point by tracing the radius from the arc center
            angle_center_to_end = angle_start_to_center + np.sign(self.__c[for_index]) * (self.__phi[for_index] - np.pi)
            this_end_point = this_arc_center + (1/abs(self.__c[for_index])) * np.array([[np.cos(angle_center_to_end),np.sin(angle_center_to_end)]], dtype=np.float32)

        # The total road length is simply the sum of the road lengths
        if (for_index == 0):
            this_l_total_at_end = self.__l[for_index:(for_index+1)]
        else:
            this_l_total_at_end = self.__l[for_index:(for_index+1)] + self.__l_total_at_end[(for_index-1):for_index]

        # Construct the hyperplace descriptions (format, Ax<=b)
        # > For the end
        this_hp_end_A = np.array([[np.cos(this_end_angle[0]),np.sin(this_end_angle[0])]], dtype=np.float32)
        this_hp_end_b = np.matmul(this_hp_end_A, np.transpose(this_end_point))[0]
        # > For the start
        this_hp_start_A = -1.0 * np.array([[np.cos(this_start_angle[0]),np.sin(this_start_angle[0])]], dtype=np.float32)
        this_hp_start_b = np.matmul(this_hp_start_A, np.transpose(this_start_point))[0]

        self.__start_points = np.concatenate((self.__start_points, this_start_point), axis=0, dtype=np.float32)
        self.__end_points   = np.concatenate((self.__end_points  , this_end_point)  , axis=0, dtype=np.float32)
        self.__start_angles = np.concatenate((self.__start_angles, this_start_angle), axis=0, dtype=np.float32)
        self.__end_angles   = np.concatenate((self.__end_angles  , this_end_angle)  , axis=0, dtype=np.float32)
        self.__arc_centers  = np.concatenate((self.__arc_centers , this_arc_center) , axis=0, dtype=np.float32)

        self.__l_total_at_end  = np.concatenate((self.__l_total_at_end , this_l_total_at_end) , axis=0, dtype=np.float32)

        self.__start_hyperplane_A  = np.concatenate((self.__start_hyperplane_A , this_hp_start_A) , axis=0, dtype=np.float32)
        self.__start_hyperplane_b  = np.concatenate((self.__start_hyperplane_b , this_hp_start_b) , axis=0, dtype=np.float32)
        self.__end_hyperplane_A  = np.concatenate((self.__end_hyperplane_A , this_hp_end_A) , axis=0, dtype=np.float32)
        self.__end_hyperplane_b  = np.concatenate((self.__end_hyperplane_b , this_hp_end_b) , axis=0, dtype=np.float32)



    def render_road(self, axis_handle):
        """
        Plot the road.

        Parameters
        ----------
            axis_handle : matplotlib.axes
                A handle for where the road is plotted.

        Returns
        -------
            plot_handles : [matplotlib.lines.Line2D]
                The handles to the lines that are plotted.
        """
        plot_handles = []
        # Iterate through the road elements
        for element_idx, this_isStraight  in enumerate(self.__isStraight):
            if (this_isStraight):
                # Directly plot the straight line
                this_handles = axis_handle.plot( [self.__start_points[element_idx,0],self.__end_points[element_idx,0]], [self.__start_points[element_idx,1],self.__end_points[element_idx,1]] )
            else:
                # Check and correct if the state and end angles cross +/- pi
                this_end_angle = self.__start_angles[element_idx] + np.sign(self.__c[element_idx]) * abs(self.__phi[element_idx])
                # Plot circle by gridding the angle range
                this_angles = -np.sign(self.__c[element_idx]) * 0.5 * np.pi + np.linspace( start=self.__start_angles[element_idx], stop=this_end_angle, num=max(2,round(0.5*abs(self.__phi[element_idx])*(180/np.pi))) )
                this_x = self.__arc_centers[element_idx,0] + (1/abs(self.__c[element_idx])) * np.cos(this_angles)
                this_y = self.__arc_centers[element_idx,1] + (1/abs(self.__c[element_idx])) * np.sin(this_angles)
                this_handles = axis_handle.plot(this_x, this_y)

            for handle in this_handles: plot_handles.append( handle )

        return plot_handles



    def find_closest_point_to(self, px, py):
        """
        For a given query point (i.e., (px,py)), this function computes the
        point on the road that is closest to that query point. Euclidean
        distance is the metric measuring and comparing distances.

        Parameters
        ----------
            px : float
                World frame x-axis coordinate of the query point (units: m).
            py : float
                World frame y-axis coordinate of the query point (units: m).

        Returns
        -------
            px_closest : float
                World frame x-axis coordinate of the closest point on the road (units: m).
            py_closest : float
                World frame y-axis coordinate of the closest point on the road (units: m).
            closest_distance : float
                Eucliedean distance between the query point and its closest point on the road (units: m).
            side_of_the_road_line : float
                The side of the road that the query point lies on (units: -1:=left-hand-side, 1:=right-hand-side)
            progress_at_closest_p : float
                The length of the whole road from its start until the closest point (units: m).
            road_angle_at_closest_p : float
                The tangent angles of the road at the closest point (units: radians).
            closest_element_idx : float
                The index of the road element that contains the closest point (units: index).
        """

        # Put the query point into a column vector
        p_query = np.array([[px],[py]],dtype=np.float32)
        # Compute the logic check for the start and end hyperplane
        start_check = np.less_equal( np.matmul(self.__start_hyperplane_A, p_query)[:,0] , self.__start_hyperplane_b )
        end_check   = np.less_equal( np.matmul(self.__end_hyperplane_A, p_query)[:,0]   , self.__end_hyperplane_b   )
        # Compute the combine check
        check = np.logical_and(start_check, end_check)
        # Get the indicies for the road elements to check
        idxs_to_check = np.where(check)[0]

        # Initialize the "closest distance" variable as this is used to keep
        # track of which point is the closest
        closest_distance = 10e10

        # Initialize the other variables that are to be returned
        px_closest = 0.0
        py_closest = 0.0
        side_of_the_road_line = 0.0
        progress_at_closest_p = 0.0
        road_angle_at_closest_p = 0.0
        closest_element_idx = 0

        # Check the start of the road
        idx_start = 0
        distance_to_start = np.sqrt((px-self.__start_points[idx_start,0])**2 + (py-self.__start_points[idx_start,1])**2)
        if (distance_to_start < closest_distance):
            # Update the closest distance and idx
            closest_distance = distance_to_start
            closest_element_idx = idx_start
            # Fill in the "easy" details
            px_closest = self.__start_points[idx_start,0]
            py_closest = self.__start_points[idx_start,1]
            progress_at_closest_p = 0.0
            road_angle_at_closest_p = self.__start_angles[idx_start]
            # Compute the side of the road
            if self.__isStraight[idx_start]:
                this_vector_to_start = p_query[:,0] - self.__start_points[idx_start,:]
                this_perp_direction = np.array([np.cos(self.__start_angles[idx_start]+0.5*np.pi),np.sin(self.__start_angles[idx_start]+0.5*np.pi)])
                this_inner_product_perp = np.inner( this_vector_to_start , this_perp_direction )
                side_of_the_road_line = (1.0) if (this_inner_product_perp >= 0) else  (-1.0)
            else:
                this_vector_to_arc_center = p_query[:,0] - self.__arc_centers[idx_start,:]
                this_dist_to_arc_center = np.linalg.norm(this_vector_to_arc_center)
                this_dist_to_arc = this_dist_to_arc_center - (1/abs(self.__c[idx_start]))
                # Compute which side of the line it is on
                if (np.sign(self.__c[idx_start]) > 0):
                    side_of_the_road_line = (1.0) if (this_dist_to_arc < 0) else  (-1.0)
                else:
                    side_of_the_road_line = (1.0) if (this_dist_to_arc > 0) else  (-1.0)



        # Check the end of the road
        idx_end = self.__c.shape[0]-1
        distance_to_end = np.sqrt((px-self.__end_points[idx_end,0])**2 + (py-self.__end_points[idx_end,1])**2)
        if (distance_to_end < closest_distance):
            # Update the closest distance and idx
            closest_distance = distance_to_end
            closest_element_idx = idx_end
            # Fill in the "easy" details
            px_closest = self.__end_points[idx_end,0]
            py_closest = self.__end_points[idx_end,1]
            progress_at_closest_p = self.__l_total_at_end[idx_end]
            road_angle_at_closest_p = self.__end_angles[idx_end]
            # Compute the side of the road
            if self.__isStraight[idx_end]:
                this_vector_to_start = p_query[:,0] - self.__start_points[idx_end,:]
                this_perp_direction = np.array([np.cos(self.__start_angles[idx_end]+0.5*np.pi),np.sin(self.__start_angles[idx_end]+0.5*np.pi)])
                this_inner_product_perp = np.inner( this_vector_to_start , this_perp_direction )
                side_of_the_road_line = (1.0) if (this_inner_product_perp >= 0) else  (-1.0)
            else:
                this_vector_to_arc_center = p_query[:,0] - self.__arc_centers[idx_end,:]
                this_dist_to_arc_center = np.linalg.norm(this_vector_to_arc_center)
                this_dist_to_arc = this_dist_to_arc_center - (1/abs(self.__c[idx_end]))
                # Compute which side of the line it is on
                if (np.sign(self.__c[idx_end]) > 0):
                    side_of_the_road_line = (1.0) if (this_dist_to_arc < 0) else  (-1.0)
                else:
                    side_of_the_road_line = (1.0) if (this_dist_to_arc > 0) else  (-1.0)



        # Check all other candidates
        for idx in idxs_to_check:
            if self.__isStraight[idx]:

                # Compute the closest distance to this road element
                this_vector_to_start = p_query[:,0] - self.__start_points[idx,:]
                this_perp_direction = np.array([np.cos(self.__start_angles[idx]+0.5*np.pi),np.sin(self.__start_angles[idx]+0.5*np.pi)])
                this_inner_product_perp = np.inner( this_vector_to_start , this_perp_direction )
                this_distance = abs( this_inner_product_perp )
                # Continue processing if this is the closest so far
                if (this_distance < closest_distance):
                    # Update the closest distance and idx
                    closest_distance = this_distance
                    closest_element_idx = idx
                    # Compute which side of the line it is on
                    side_of_the_road_line = (1.0) if (this_inner_product_perp >= 0) else  (-1.0)
                    # Compute the point to return
                    distance_from_start = np.inner( this_vector_to_start , self.__end_hyperplane_A[idx,:] )
                    px_closest = self.__start_points[idx,0] + distance_from_start * np.cos(self.__start_angles[idx])
                    py_closest = self.__start_points[idx,1] + distance_from_start * np.sin(self.__start_angles[idx])
                    # Compute the progress of the closest point
                    progress_at_closest_p = self.__l_total_at_end[idx] - self.__l[idx] + distance_from_start
                    # Set the road angle at this closest point
                    road_angle_at_closest_p = self.__start_angles[idx]
            else:
                # Compute the closest distance to this road element
                this_vector_to_arc_center = p_query[:,0] - self.__arc_centers[idx,:]
                this_dist_to_arc_center = np.linalg.norm(this_vector_to_arc_center)
                this_dist_to_arc = this_dist_to_arc_center - (1/abs(self.__c[idx]))
                this_distance = abs( this_dist_to_arc )
                # Continue processing if this is the closest so far
                if (this_distance < closest_distance):
                    # Update the closest distance and idx
                    closest_distance = this_distance
                    closest_element_idx = idx
                    # Compute which side of the line it is on
                    if (np.sign(self.__c[idx]) > 0):
                        side_of_the_road_line = (1.0) if (this_dist_to_arc < 0) else  (-1.0)
                    else:
                        side_of_the_road_line = (1.0) if (this_dist_to_arc > 0) else  (-1.0)
                    # Compute the point to return
                    p_closest = self.__arc_centers[idx,:] + this_vector_to_arc_center * 1/(this_dist_to_arc_center*abs(self.__c[idx]))
                    px_closest = p_closest[0]
                    py_closest = p_closest[1]
                    # Compute the chord length from the start to the closest point
                    this_chord_vector_start_to_closest_p = p_closest - self.__start_points[idx,:]
                    this_chord_length_start_to_closest_p = np.linalg.norm(this_chord_vector_start_to_closest_p)
                    # Compute the angle from the start to the closest p
                    this_angle_start_to_closest_p = 2.0 * np.arcsin(0.5*this_chord_length_start_to_closest_p*abs(self.__c[idx]))
                    # Compute the progress of the closest point
                    progress_at_closest_p = self.__l_total_at_end[idx] - self.__l[idx] + (this_angle_start_to_closest_p/abs(self.__c[idx]))
                    # Set the road angle at this closest point
                    if (np.sign(self.__c[idx]) > 0):
                        road_angle_at_closest_p = self.__start_angles[idx] + this_angle_start_to_closest_p
                    else:
                        road_angle_at_closest_p = self.__start_angles[idx] - this_angle_start_to_closest_p

        # Return the results
        return px_closest, py_closest, closest_distance, side_of_the_road_line, progress_at_closest_p, road_angle_at_closest_p, closest_element_idx



    def zzz_old_find_closest_point_to(self, px, py):
        """
        For a given query point (i.e., (px,py)), this function computes the
        point on the road that is closest to that query point. Euclidean
        distance is the metric measuring and comparing distances.

        Parameters
        ----------
            px : float
                World frame x-axis coordinate of the query point (units: m).
            py : float
                World frame y-axis coordinate of the query point (units: m).

        Returns
        -------
            px_closest : float
                World frame x-axis coordinate of the closest point on the road (units: m).
            py_closest : float
                World frame y-axis coordinate of the closest point on the road (units: m).
            closest_distance : float
                Eucliedean distance between the query point and its closest point on the road (units: m).
            side_of_the_road_line : float
                The side of the road that the query point lies on (units: -1:=left-hand-side, 1:=right-hand-side)
            progress_at_closest_p : float
                The length of the whole road from its start until the closest point (units: m).
            road_angle_at_closest_p : float
                The tangent angles of the road at the closest point (units: radians).
            closest_element_idx : float
                The index of the road element that contains the closest point (units: index).
        """

        # Put the query point into a column vector
        p_query = np.array([[px],[py]],dtype=np.float32)
        # Compute the logic check for the start and end hyperplane
        start_check = np.less_equal( np.matmul(self.__start_hyperplane_A, p_query)[:,0] , self.__start_hyperplane_b )
        end_check   = np.less_equal( np.matmul(self.__end_hyperplane_A, p_query)[:,0]   , self.__end_hyperplane_b   )
        # Compute the combine check
        check = np.logical_and(start_check, end_check)
        # Get the indicies for the road elements to check
        idxs_to_check = np.where(check)[0]

        # Initialize the "closest distance" variable as this is used to keep
        # track of which point is the closest
        closest_distance = 10e10

        # Initialize the other variables that are to be returned
        px_closest = 0.0
        py_closest = 0.0
        side_of_the_road_line = 0.0
        progress_at_closest_p = 0.0
        road_angle_at_closest_p = 0.0
        closest_element_idx = 0

        # Check the start of the road
        idx_start = 0
        distance_to_start = np.sqrt((px-self.__start_points[idx_start,0])**2 + (py-self.__start_points[idx_start,1])**2)
        if (distance_to_start < closest_distance):
            # Update the closest distance and idx
            closest_distance = distance_to_start
            closest_element_idx = idx_start
            # Fill in the "easy" details
            px_closest = self.__start_points[idx_start,0]
            py_closest = self.__start_points[idx_start,1]
            progress_at_closest_p = 0.0
            road_angle_at_closest_p = self.__start_angles[idx_start]
            # Compute the side of the road
            if self.__isStraight[idx_start]:
                this_vector_to_start = p_query[:,0] - self.__start_points[idx_start,:]
                this_perp_direction = np.array([np.cos(self.__start_angles[idx_start]+0.5*np.pi),np.sin(self.__start_angles[idx_start]+0.5*np.pi)])
                this_inner_product_perp = np.inner( this_vector_to_start , this_perp_direction )
                side_of_the_road_line = (1.0) if (this_inner_product_perp >= 0) else  (-1.0)
            else:
                this_vector_to_arc_center = p_query[:,0] - self.__arc_centers[idx_start,:]
                this_dist_to_arc_center = np.linalg.norm(this_vector_to_arc_center)
                this_dist_to_arc = this_dist_to_arc_center - (1/abs(self.__c[idx_start]))
                # Compute which side of the line it is on
                if (np.sign(self.__c[idx_start]) > 0):
                    side_of_the_road_line = (1.0) if (this_dist_to_arc < 0) else  (-1.0)
                else:
                    side_of_the_road_line = (1.0) if (this_dist_to_arc > 0) else  (-1.0)



        # Check the end of the road
        idx_end = self.__c.shape[0]-1
        distance_to_end = np.sqrt((px-self.__end_points[idx_end,0])**2 + (py-self.__end_points[idx_end,1])**2)
        if (distance_to_end < closest_distance):
            # Update the closest distance and idx
            closest_distance = distance_to_end
            closest_element_idx = idx_end
            # Fill in the "easy" details
            px_closest = self.__end_points[idx_end,0]
            py_closest = self.__end_points[idx_end,1]
            progress_at_closest_p = self.__l_total_at_end[idx_end]
            road_angle_at_closest_p = self.__end_angles[idx_end]
            # Compute the side of the road
            if self.__isStraight[idx_end]:
                this_vector_to_start = p_query[:,0] - self.__start_points[idx_end,:]
                this_perp_direction = np.array([np.cos(self.__start_angles[idx_end]+0.5*np.pi),np.sin(self.__start_angles[idx_end]+0.5*np.pi)])
                this_inner_product_perp = np.inner( this_vector_to_start , this_perp_direction )
                side_of_the_road_line = (1.0) if (this_inner_product_perp >= 0) else  (-1.0)
            else:
                this_vector_to_arc_center = p_query[:,0] - self.__arc_centers[idx_end,:]
                this_dist_to_arc_center = np.linalg.norm(this_vector_to_arc_center)
                this_dist_to_arc = this_dist_to_arc_center - (1/abs(self.__c[idx_end]))
                # Compute which side of the line it is on
                if (np.sign(self.__c[idx_end]) > 0):
                    side_of_the_road_line = (1.0) if (this_dist_to_arc < 0) else  (-1.0)
                else:
                    side_of_the_road_line = (1.0) if (this_dist_to_arc > 0) else  (-1.0)



        # Check all other canditates
        for idx in idxs_to_check:
            if self.__isStraight[idx]:

                # Compute the closest distance to this road element
                this_vector_to_start = p_query[:,0] - self.__start_points[idx,:]
                this_perp_direction = np.array([np.cos(self.__start_angles[idx]+0.5*np.pi),np.sin(self.__start_angles[idx]+0.5*np.pi)])
                this_inner_product_perp = np.inner( this_vector_to_start , this_perp_direction )
                this_distance = abs( this_inner_product_perp )
                # Continue processing if this is the closest so far
                if (this_distance < closest_distance):
                    # Update the closest distance and idx
                    closest_distance = this_distance
                    closest_element_idx = idx
                    # Compute which side of the line it is on
                    side_of_the_road_line = (1.0) if (this_inner_product_perp >= 0) else  (-1.0)
                    # Compute the point to return
                    distance_from_start = np.inner( this_vector_to_start , self.__end_hyperplane_A[idx,:] )
                    px_closest = self.__start_points[idx,0] + distance_from_start * np.cos(self.__start_angles[idx])
                    py_closest = self.__start_points[idx,1] + distance_from_start * np.sin(self.__start_angles[idx])
                    # Compute the progress of the closest point
                    progress_at_closest_p = self.__l_total_at_end[idx] - self.__l[idx] + distance_from_start
                    # Set the road angle at this closest point
                    road_angle_at_closest_p = self.__start_angles[idx]
            else:
                # Compute the closest distance to this road element
                this_vector_to_arc_center = p_query[:,0] - self.__arc_centers[idx,:]
                this_dist_to_arc_center = np.linalg.norm(this_vector_to_arc_center)
                this_dist_to_arc = this_dist_to_arc_center - (1/abs(self.__c[idx]))
                this_distance = abs( this_dist_to_arc )
                # Continue processing if this is the closest so far
                if (this_distance < closest_distance):
                    # Update the closest distance and idx
                    closest_distance = this_distance
                    closest_element_idx = idx
                    # Compute which side of the line it is on
                    if (np.sign(self.__c[idx]) > 0):
                        side_of_the_road_line = (1.0) if (this_dist_to_arc < 0) else  (-1.0)
                    else:
                        side_of_the_road_line = (1.0) if (this_dist_to_arc > 0) else  (-1.0)
                    # Compute the point to return
                    p_closest = self.__arc_centers[idx,:] + this_vector_to_arc_center * 1/(this_dist_to_arc_center*abs(self.__c[idx]))
                    px_closest = p_closest[0]
                    py_closest = p_closest[1]
                    # Compute the progress of the closest point
                    angle_of_center_to_query_vector = np.arctan2( this_vector_to_arc_center[1], this_vector_to_arc_center[0] )
                    vector_arc_center_to_arc_start = self.__start_points[idx,:] - self.__arc_centers[idx,:]
                    angle_of_center_to_arc_start = np.arctan2( vector_arc_center_to_arc_start[1], vector_arc_center_to_arc_start[0] )
                    progress_at_closest_p = self.__l_total_at_end[idx] - self.__l[idx] + (1/abs(self.__c[idx])) * abs(angle_of_center_to_query_vector-angle_of_center_to_arc_start)
                    # Set the road angle at this closest point
                    proportion_progress = abs(angle_of_center_to_query_vector-angle_of_center_to_arc_start) / self.__phi[idx]
                    road_angle_at_closest_p = self.__start_angles[idx] + proportion_progress * (self.__end_angles[idx]-self.__start_angles[idx])

        # Return the results
        return px_closest, py_closest, closest_distance, side_of_the_road_line, progress_at_closest_p, road_angle_at_closest_p, closest_element_idx



    def convert_progression_to_coordinates(self, progression_queries):
        """
        Computes the world-frame coordinates of points along the road where the length of
        whole road (i.e., progression from the start of the road) equals to the query value(s).

        Parameters
        ----------
            progressing query : numpy array
                Value of road progression for which the world-frame coordinates. should be computed (units: m).

        Returns
        -------
            p_coords : numpy array
                The world frame coordinates for each progression query,
                2-dimensional array with size = number of queries -by- 2,
                (units: m).
            p_angles : numpy array
                The tangent angle of the road for each progression query,
                1-dimensional array with length = number of queries,
                (units: radians).
        """
        # Get the number of progression query values
        num_queries = len(progression_queries)

        # Initialise arrays for the results to be returned
        p_coords = np.zeros((num_queries,2), dtype=np.float32)
        p_angles = np.zeros((num_queries,), dtype=np.float32)

        # Compute the index of the road element for each progression value
        road_idxs = np.searchsorted( self.__l_total_at_end , progression_queries, side="left", sorter=None)

        # Iterate through the progression values
        for i_prog in np.arange(0,num_queries):
            # Get the value of progression
            this_prog = progression_queries[i_prog]

            # Get the road element index
            this_road_idx = road_idxs[i_prog]

            # Check if the progression is beyond the end of the road
            if (this_road_idx == len(self.__l_total_at_end) ):
                p_coords[i_prog,0] = np.nan
                p_coords[i_prog,1] = np.nan
                p_angles[i_prog]   = np.nan

            # Also check if the progression is before the start of the road
            elif (this_prog < 0):
                p_coords[i_prog,0] = np.nan
                p_coords[i_prog,1] = np.nan
                p_angles[i_prog]   = np.nan

            # Otherwise:
            else:
                # Compute the extra progression from the start of this road element
                this_prog_from_start_of_road_element = this_prog - (self.__l_total_at_end[this_road_idx]-self.__l[this_road_idx])

                # Switch depending on the type of road element
                if (self.__isStraight[this_road_idx]):
                    # Get the angle
                    this_angle = self.__start_angles[this_road_idx]
                    # Compute the coords
                    p_coords[i_prog,0] = self.__start_points[this_road_idx,0] + this_prog_from_start_of_road_element * np.cos(this_angle)
                    p_coords[i_prog,1] = self.__start_points[this_road_idx,1] + this_prog_from_start_of_road_element * np.sin(this_angle)
                    # Put in the angle
                    p_angles[i_prog] = this_angle

                else:
                    # Compute the angular range from the start to the progress point
                    this_prog_phi = (this_prog_from_start_of_road_element/self.__l[this_road_idx]) * self.__phi[this_road_idx]
                    # Compute the chord length from the start to the progress point
                    this_chord_length = 2.0 * (1/abs(self.__c[this_road_idx])) * np.sin(0.5*this_prog_phi)
                    # Compute the angle of the chord
                    this_chord_angle = self.__start_angles[this_road_idx] + np.sign(self.__c[this_road_idx]) * 0.5 * this_prog_phi
                    # Compute the coords
                    p_coords[i_prog,0] = self.__start_points[this_road_idx,0] + this_chord_length * np.cos(this_chord_angle)
                    p_coords[i_prog,1] = self.__start_points[this_road_idx,1] + this_chord_length * np.sin(this_chord_angle)
                    # Compute the angle
                    p_angles[i_prog] = this_chord_angle + np.sign(self.__c[this_road_idx]) * 0.5 * this_prog_phi

        # Return the coordinates and angles
        return p_coords, p_angles



    def convert_progression_to_curvature(self, progression_queries):
        """
        Retrieves the curvature at points along the road where the length of
        whole road (i.e., progression from the start of the road) equals to the query value(s).

        Parameters
        ----------
            progressing query : numpy array
                Value of road progression for which the world-frame coordinates. should be computed (units: m).

        Returns
        -------
            p_curvatures : numpy array
                The curvature of the road for each progression query,
                1-dimensional array with length = number of queries,
                (units: 1/m).
        """
        # Get the number of progression query values
        num_queries = len(progression_queries)

        # Initialise arrays for the results to be returned
        p_curvatures = np.zeros((num_queries,), dtype=np.float32)

        # Compute the index of the road element for each progression value
        road_idxs = np.searchsorted( self.__l_total_at_end , progression_queries, side="left", sorter=None)

        # Iterate through the progression values
        for i_prog in np.arange(0,num_queries):
            # Get the value of progression
            this_prog = progression_queries[i_prog]

            # Get the road element index
            this_road_idx = road_idxs[i_prog]

            # Check if the progression is beyond the end of the road
            if (this_road_idx == len(self.__l_total_at_end) ):
                p_curvatures[i_prog] = np.nan

            # Also check if the progression is before the start of the road
            elif (this_prog < 0):
                p_curvatures[i_prog] = np.nan

            # Otherwise:
            else:
                # Extract the curvature
                p_curvatures[i_prog] = self.__c[this_road_idx]

        # Return the coordinates and angles
        return p_curvatures



    @staticmethod
    def transform_points_2d( p , px_translate, py_translate, theta_rotate, p_translate_is_in_starting_frame=True):
        """
        Transform 2D points from one frame to another. The order of translate
        and rotate depdned on which frame the translation vector is given in.

        If the transalation vector is expressed in the coordinates of the
        original frame (i.e., the untransformed frame), then the steps are:
            1. Translate by (px_translate,py_translate).
            2. Rotate by theta_rotate.

        If the transalation vector is expressed in the coordinates of the
        destination frame (i.e., the transformed frame), then the steps are:
            1. Rotate by theta_rotate.
            2. Translate by (px_translate,py_translate).

        Parameters
        ----------
            p : numpy array
                The point to be transformed, 2-dimensional array with:
                size = number of points to transform -by- 2,
                (units: m).
            px_translate : float
                X-axis length of the translation vector (units: m).
            py_translate : float
                Y-axis length of the translation vector (units: m).
            theta_rotate : float
                The angle of rotation between the original and destination
                frames (units: radians).
            p_translate_is_in_starting_frame : boolean
                A flag for indicating the frame of the translation vector:
                True := translation vector is in the original frame.
                True := translation vector is in the destination frame.

        Returns
        -------
            p_transformed : numpy array
                The coordinates for each point in "p" expressed in coordinates
                of the destination frame, 2-dimensional array with:
                size = number of points to transform -by- 2,
                (units: m).
        """
        # Construct the rotation matrix
        R_mat = np.array([[np.cos(theta_rotate),np.sin(theta_rotate)],[-np.sin(theta_rotate),np.cos(theta_rotate)]])
        if (p_translate_is_in_starting_frame):
            # Translate all the point
            p_translated = np.subtract( p , np.array([[px_translate,py_translate]]) )
            # Rotate all the points
            p_transformed = np.matmul( R_mat , np.transpose(p_translated) )
        else:
            # Rotate all the points
            p_rotated = np.matmul( R_mat , np.transpose(p) )
            # Translate all the point
            p_transformed = np.subtract( p_rotated , np.array([[px_translate],[py_translate]]) )

        # Return the result
        return np.transpose(p_transformed)



    def road_info_at_given_pose_and_progress_queries(self, px, py, theta, progress_queries):
        """
        Utilizes the other functions in this class to:
        1. Find the closest point on the road for the given query point (px,py)
        2. Generate a sequence of progression values forwards (and possibly
           even backwards) from that closest point, i.e., the "progress_queries"
        3. Get the world-frame coordinates for all the progression values.

        Parameters
        ----------
            px : float
                World-frame x-axis coordinate of the robot / car (units: m).
            py : float
                World-frame y-axis coordinate of the robot / car (units: m).
            theta : float
                Heading angle of the robot / car relative to the world-frame
                x-axis (units: radians).
            progress_queries : numpy arrray, 1-dimensional
                Specifies the values of progress-along-the-road, relative to
                the current position of the car, at which the road information
                should be generated.
                Units: meters

        Returns
        -------
            info_dict : dictionary
                Containing details for the road relative to the current
                state of the car.
                The properties of the info_dict are:
                - "px", "py" : float
                    World-frame (x,y) coordinate of the car.
                - "px_closest", "py_closest" : float
                    World-frame (x,y) coordinate of the closest point on the road.
                - "closest_distance" : float
                    Euclidean distance from the car to the closest point on the road.
                - "side_of_the_road_line" : int
                    The side of the road that the car is on (1:=left, -1=right).
                - "progress_at_closest_p" : float
                    The total length of road from the start of the road to the closest point.
                - "road_angle_at_closest_p" : float
                    The angle of the road at the closest point (relative to the world-frame x-axis).
                - "road_angle_relative_to_body_frame_at_closest_p" : float
                    Angle of the road, relative to the body frame, at the closest point.
                - "curvature_at_closest_p" : float
                    The curvature of the road at the closest point.
                - "closest_element_idx" : int
                    The index of the road element that is closest to the car.
                - "progress_queries" : numpy array, 1-dimensional
                    A repeat of the input parameter that specifies the values of progress-along-the-road,
                    relative to the current position of the car, at which the observations should be generated. 
                - "road_points_in_body_frame" : numpy array, 2-dimensional
                    (px,py) coordinates in the body frame of the progress query points.
                    A 2-dimensional numpy array with: size = number of query points -by- 2.
                - "road_angles_relative_to_body_frame" : numpy array, 1-dimensional
                    Angle of the road, relative to the body frame, at each of the progress query points.
                    A 1-dimensional numpy array with: size = number of query points.
                - "curvatures" : numpy array, 1-dimensional
                    Curvature of the road at each of the progress query points.
                    A 1-dimensional numpy array with: size = number of query points.

                Units: all length in meters, all angles in radians.
        """
        # Compute the closest point on the road
        px_closest, py_closest, closest_distance, side_of_the_road_line, progress_at_closest_p, road_angle_at_closest_p, closest_element_idx = self.find_closest_point_to(px=px, py=py)

        # Compute the relative road angle at the closest point
        road_angle_relative_to_body_frame_at_closest_p = road_angle_at_closest_p - theta

        # Get the curvature of the closest element
        curvature_at_closest_p = self.__c[closest_element_idx]

        # Shift the progression queries by the progression of the current point
        prog_queries_from_start = progress_at_closest_p + progress_queries

        # Convert the progression array to coordinates and angles along the upcoming section of road
        p_coords, p_angles = self.convert_progression_to_coordinates(prog_queries_from_start)

        # Transform the points into the body frame of the given pose
        p_coords_in_body_frame = Road.transform_points_2d( p_coords , px, py, theta)

        # Shift the angles to be relative to the body frame of the given pose
        p_angles_relative_to_body_frame = p_angles - theta

        # Get the curvature at the progression queries
        p_curvatures = self.convert_progression_to_curvature(prog_queries_from_start)

        # Create an info dictionary with all the extra details
        info_dict = {
            "px" : px,
            "py" : py,
            "px_closest" : px_closest,
            "py_closest" : py_closest,
            "closest_distance" : closest_distance,
            "side_of_the_road_line" : side_of_the_road_line,
            "progress_at_closest_p" : progress_at_closest_p,
            "road_angle_at_closest_p" : road_angle_at_closest_p,
            "road_angle_relative_to_body_frame_at_closest_p" : road_angle_relative_to_body_frame_at_closest_p,
            "curvature_at_closest_p" : curvature_at_closest_p,
            "closest_element_idx" : closest_element_idx,
            "progress_queries" : progress_queries,
            "road_points_in_body_frame" : p_coords_in_body_frame,
            "road_angles_relative_to_body_frame" : p_angles_relative_to_body_frame,
            "curvatures" : p_curvatures,
        }

        # Return the pixel coordinates and the info dictionary
        return info_dict


    @staticmethod
    def transform_to_camera_pixel( p , R_BtoC, T_CtoB_inB, intrinsic_matrix):
        """
        Transform 3-dimensional coordinates expressed in the body frame of the
        robot / car, to 2-dimensional pixel coordinates of the camera that is
        observing the world environment from its fixed mounting location on the
        robot / car.

        Parameters
        ----------
            p : numpy array
                The points to be transformed, 2-dimensional array with:
                size = number of points to transform -by- 3,
                (units: m).
            R_BtoC : numpy array
                The rotation matrix from the body frame to the camera frame,
                2-dimensional array with: size = 3 -by- 3, (units: unitless).
            T_CtoB_inB : numpy array
                Translation vector that points from the origin on the camera
                frame (i.e., the center point of the camer lens), to the origin
                of the body frame, 2-dimensional array with: size = 3 -by- 1,
                (units: m).
            intrinsic_matrix : numpy array
                The matrix that contain the intrinsic paramters of the camera
                in the "usual" format of: [ [fx 0 cx] , [0 fy cy] , [0 0 1] ],
                2-dimensional array with: size = 3 -by- 3, (units: pixels/m).

        Returns
        -------
            uv : numpy array
                The coordinates for each point in "p" expressed in coordinates
                of the camera pixels, "u" is the pixel location in the
                camera-frame x-direction, and "v" in the camera-frame
                y-direction, 2-dimensional array with:
                size = number of points to transform -by- 2,
                (units: pixels).
        """
        # Convert all the points to the scaled pixel coordinates
        uv_scaled = np.matmul( intrinsic_matrix , np.matmul( R_BtoC , np.transpose(p) + T_CtoB_inB ) )

        # Disregard points that are behind (or too close to) the camera
        uv_lambda = uv_scaled[2:3,:]
        uv_lambda[uv_lambda<0.01] = np.nan

        uv = np.divide( uv_scaled[0:2,:] , uv_lambda )

        # Return the result
        return np.transpose(uv)



    def line_detection_pixels_for_given_pose(self, px, py, theta, R_BtoC, T_CtoB_inB, intrinsic_matrix, resolution_width, resolution_height ):
        """
        Utilizes the other functions in this class to minic the output of a
        line detection algorithm running on a camera image captured by the
        robot / car. The major steps of this function are:
        1. Find the closest point on the road for the given query point (px,py)
        2. Generate a sequence of progression values forwards and backwards from
           that closest point.
        3. Get the world-frame coordinates for all the progression values.
        4. Transform those world-frame coordinates into pixel coordinates.

        Parameters
        ----------
            px : float
                World-frame x-axis coordinate of the robot / car (units: m).
            py : float
                World-frame y-axis coordinate of the robot / car (units: m).
            theta : float
                Heading angle of the robot / car relative to the world-frame
                x-axis (units: radians).
            R_BtoC : numpy array
                The rotation matrix from the body frame to the camera frame,
                2-dimensional array with: size = 3 -by- 3, (units: unit-less).
            T_CtoB_inB : numpy array
                Translation vector that points from the origin on the camera
                frame (i.e., the center point of the camera lens), to the origin
                of the body frame, 2-dimensional array with: size = 3 -by- 1,
                (units: m).
            intrinsic_matrix : numpy array
                The matrix that contain the intrinsic parameters of the camera
                in the "usual" format of: [ [fx 0 cx] , [0 fy cy] , [0 0 1] ],
                2-dimensional array with: size = 3 -by- 3, (units: pixels/m).
            resolution_width : integer
                The pixel size of the camera's pixel array in the camera-frame
                x-axis direction
            resolution_height : integer
                The pixel size of the camera's pixel array in the camera-frame
                y-axis direction

        Returns
        -------
            uv_coords : numpy array
                The coordinates for each line detection point expressed in
                coordinates of the camera pixels, "u" is the pixel location in
                the camera-frame x-direction, and "v" in the camera-frame
                y-direction, 2-dimensional array with:
                size = number of points detected within the camera's pixel array -by- 2,
                (units: pixels).
            info_dict : dictionary
                Containing additional detail for each of the points in the
                "uv_coords" array.
        """
        # Compute the closest point on the road
        px_closest, py_closest, closest_distance, side_of_the_road_line, progress_at_closest_p, road_angle_at_closest_p, closest_element_idx = self.find_closest_point_to(px=px, py=py)

        # Construct an array of progression for a sufficiently long look ahead
        l_total_at_end = self.get_l_total_at_end()
        #prog_queries = np.arange(start=max(0,progress_at_closest_p-2), stop=progress_at_closest_p+50, step=0.5, dtype=np.float32)
        #prog_queries = np.logspace(start=np.log10(max(0.01,progress_at_closest_p-2)), stop=np.log10(progress_at_closest_p+50), num=100, base=10, dtype=np.float32)
        prog_queries = np.concatenate((
            np.arange(start=max(0,progress_at_closest_p-2), stop=progress_at_closest_p+2, step=0.1, dtype=np.float32),
            np.arange(start=progress_at_closest_p+2, stop=progress_at_closest_p+7, step=0.2, dtype=np.float32),
            np.arange(start=progress_at_closest_p+7, stop=progress_at_closest_p+20, step=0.5, dtype=np.float32),
            np.arange(start=progress_at_closest_p+20, stop=progress_at_closest_p+30, step=1.0, dtype=np.float32),
            np.arange(start=progress_at_closest_p+30, stop=progress_at_closest_p+50, step=2.0, dtype=np.float32),
        ))


        # Convert the progession array to coordinates and angles along the upcoming section of road
        p_coords, p_angles = self.convert_progression_to_coordinates(prog_queries)

        # Transform the points into the body frame of the given pose
        p_in_body_frame = Road.transform_points_2d( p_coords , px, py, theta)

        # Add the body-frame z-coordinates to be zero, i.e., the 2D ground plane
        p_3D_in_body_frame = np.hstack( (p_in_body_frame , np.zeros((p_in_body_frame.shape[0],1),dtype=np.float32) ))

        # Compute the pixel cordinates
        uv_coords_all = self.transform_to_camera_pixel( p_3D_in_body_frame , R_BtoC, T_CtoB_inB, intrinsic_matrix)

        # Remove coordinates that are not in the image array
        pixel_buffer = 0
        valid_uv_indices = np.logical_and( np.logical_and(uv_coords_all[:,0]>=(0-pixel_buffer), uv_coords_all[:,0]<=(resolution_width+pixel_buffer)) , np.logical_and(uv_coords_all[:,1]>=(0-pixel_buffer), uv_coords_all[:,1]<=(resolution_height+pixel_buffer)) )
        uv_coords = uv_coords_all[valid_uv_indices,:]

        p_visible_in_body_frame = p_in_body_frame[valid_uv_indices,:]

        progression_of_each_coord = prog_queries[valid_uv_indices]

        # Create an info dictionary with all the extra details
        info_dict = {
            "px" : px,
            "py" : py,
            "px_closest" : px_closest,
            "py_closest" : py_closest,
            "closest_distance" : closest_distance,
            "side_of_the_road_line" : side_of_the_road_line,
            "progress_at_closest_p" : progress_at_closest_p,
            "road_angle_at_closest_p" : road_angle_at_closest_p,
            "closest_element_idx" : closest_element_idx,
            "p_visible_in_body_frame" : p_visible_in_body_frame,
            "progression_of_each_coord" : progression_of_each_coord,
        }

        # Return the pixel coordinates and the info dictionary
        return uv_coords, info_dict