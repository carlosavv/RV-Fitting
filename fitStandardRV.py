# imports
from geomdl import fitting
from geomdl import convert


def fit_Standard_RV(N,sampled_data):

	# set up the fitting parameters
	p_ctrlpts = sampled_data
	size_u = N+1
	size_v = N+1
	degree_u = 3
	degree_v = 3

	# Do global surface approximation
	standard_RV_NURBS_surf = fitting.interpolate_surface(p_ctrlpts, size_u, size_v, degree_u, degree_v)

	standard_RV_NURBS_surf = convert.bspline_to_nurbs(standard_RV_NURBS_surf)

	return standard_RV_NURBS_surf

