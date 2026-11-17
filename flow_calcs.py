import numpy as np
import math
import matplotlib.pyplot as plt

###Major Variables
#piston_vel = 15 #in/s
#needle_1_displacement = 0.075
#needle_2_displacement = 0.05

###System Setup	
piston_displacement = 4 #in (0,8)
ext_reservoir_preload = 200 #(lbs/in^2)
fluid_density = 0.0000796 #(lbf*s^2/in^4)	
dynamic_viscosity = 0.00000345 #(lbf*s/in^2)
surface_roughness = 0.000125 #(in) #check these units #note that this is just a globally defined value for computing Re
	
###Exhaust Geo	
exh_hole_ID = 0.098
num_exh_holes = 12
washer_ID = 0.866 
installed_length = 0.4
exh_flow_area_load = ((np.pi*exh_hole_ID**2)/4)*num_exh_holes
#print("exh flow area load: ", exh_flow_area_load)
	
###Shim Geo
shim_hole_ID = 0.098
num_shim_holes = 8
big_shim_OD = 1
shim_max_deflection = 0.15
cracking_pressure = 2
shim_flow_area_load = ((np.pi*shim_hole_ID**2)/4)*num_shim_holes
delta_h_max = 0.1
#print("shim flow area load: ", shim_flow_area_load)
    
###Needle Geo	
dim_1 = 0.2
theta = 45
d = dim_1/(2*np.tan(np.deg2rad(theta)))
#print("needle 1 area: ", needle_1_area)
#print("needle 2 area: ", needle_2_area)
    
###Piston Geo	
bore_ID = 1.17
bore_area = (np.pi*bore_ID**2)/4
shaft_OD = 0.25
shaft_area = (np.pi*shaft_OD**2)/4
inner_twin_length = 8.5 
piston_head_length = 0.5
outer_twin_length = 9
outer_twin_ID = 1.25
outer_twin_OD = 1.75 
inner_twin_BS_joint_approx_ID = 1
outer_twin_BS_joint_approx_ID = 1
outer_twin_area = (np.pi/4)*(outer_twin_OD**2-outer_twin_ID**2)
outer_twin_hydraulic_diameter = 4*outer_twin_area/(np.pi*(outer_twin_OD+outer_twin_ID))
inner_twin_HS_hydraulic_diameter = 4*(bore_area-shaft_area)/(np.pi*(bore_ID+shaft_OD))
#print("outer twin area: ", outer_twin_area)
#print("outer twin hyd dia: ", outer_twin_hydraulic_diameter)
#print("inner twin HS hyd dia: ", inner_twin_HS_hydraulic_diameter)


###Head Hole Geo	
head_hole_ID = 0.3
num_head_holes = 24
head_hole_flow_area = ((np.pi*head_hole_ID**2)/4)*num_head_holes
#print("head hole flow area: ", head_hole_flow_area)

###Spring Stats
free_length = 0.4
spring_rate = 2
spring_crack_pressure = (free_length-installed_length)*spring_rate/exh_flow_area_load
#print("spring crack pressure: ", spring_crack_pressure)

###calculate needle area as a function of x
def needle_area(x, theta=theta, dim_1=dim_1, d=d):
    area = np.pi*((dim_1**2/4) - ((d-x)*np.tan(np.deg2rad(theta)))**2)
    return area

###Shim Needle Interactions
def newton_method_shim_needle_calc(prescribed_flow, needle_area, needle_loss_coeff, shim_dis_coeff, shim_max_def=shim_max_deflection, shim_flow_area_load=shim_flow_area_load, big_shim_OD=big_shim_OD, starting_displacement=0, delta_h_max=delta_h_max, fluid_density=fluid_density):
    h = starting_displacement
    K1 = 100
    K2 = 10
    K3 = 1
    K4 = 0.12069
    a = shim_dis_coeff*np.pi*big_shim_OD
    b = needle_loss_coeff*needle_area
    shim_crack_pressure = K4 / shim_flow_area_load
    Q_crack = b * np.sqrt(2 * shim_crack_pressure / fluid_density)
    if prescribed_flow <= Q_crack:
        return (fluid_density / 2) * (prescribed_flow / b)**2

    for iteration in range(100):
        force = K1*h**3 + K2*h**2 + K3*h + K4
        force_prime = 3*K1*h**2 + 2*K2*h + K3
        delta_p = force/shim_flow_area_load
        shim_flow_area = np.pi*big_shim_OD*h
        flow_rate = (shim_dis_coeff*shim_flow_area+needle_loss_coeff*needle_area)*np.sqrt(2*delta_p/fluid_density)
        error = prescribed_flow-flow_rate
        c = np.sqrt(2/(fluid_density*shim_flow_area_load))
        flow_rate_prime = a*c*np.sqrt(force)+(a*c*h+b*c)*force_prime/(2*np.sqrt(force))
        if abs(error/flow_rate_prime) <= delta_h_max:
            delta_h = error/flow_rate_prime
        else:
            delta_h = (delta_h_max)*(error/flow_rate_prime)/abs(error/flow_rate_prime)
        if abs(error) < 10**-6:
            if h > shim_max_def:
                delta_p = (fluid_density / 2) * (prescribed_flow / (a*shim_max_def+b))**2
            break
        h = h + delta_h
    else:
        raise RuntimeError("shim calc did not converge")
    return delta_p

###Exhaust Needle Interactions       
def direct_solve_exhaust_needle_calc(prescribed_flow, needle_area, needle_loss_coeff, exh_dis_coeff, fluid_density=fluid_density, washer_ID=washer_ID, exh_flow_area_load=exh_flow_area_load, spring_rate=spring_rate, spring_crack_pressure=spring_crack_pressure):
    Q = prescribed_flow
    A = np.sqrt(2/fluid_density)*(exh_dis_coeff*np.pi*washer_ID*exh_flow_area_load/spring_rate)
    B = np.sqrt(2/fluid_density)*(needle_loss_coeff*needle_area - (exh_dis_coeff*np.pi*washer_ID*exh_flow_area_load/spring_rate)*spring_crack_pressure)
    E = Q/(2*A)
    F = Q**2/(4*A**2)
    G = B**3/(27*A**3)
    x = math.cbrt(E + np.sqrt(F+G)) + math.cbrt(E - np.sqrt(F+G))
    delta_p = x**2
    exh_deflection = (delta_p-spring_crack_pressure)*exh_flow_area_load/spring_rate
    if delta_p <= spring_crack_pressure:
        delta_p = (fluid_density/2)*(prescribed_flow/(needle_loss_coeff*needle_area))**2
        exh_deflection = 0
    exh_flow_area = np.pi*exh_deflection*washer_ID
    return delta_p

###Joint Pressure Losses
def joint_orifice_calc(prescribed_flow, area, joint_loss_coeff, fluid_density=fluid_density):
    delta_p = (fluid_density/2)*(prescribed_flow/(area*joint_loss_coeff))**2
    return delta_p

###friction factor calc
def reynolds_ff(fluid_vel, hyd_diameter, surface_roughness=surface_roughness, density=fluid_density, viscosiy=dynamic_viscosity):
    Re = density*fluid_vel*hyd_diameter/viscosiy
    if Re < 4000:
        friction_factor = 64/Re
    else:
        friction_factor = 1/(-1.8*math.log10((surface_roughness/(3.7*hyd_diameter))**1.11 + 6.9/Re))**2
    return friction_factor

###Major losses in pipe sections
def major_loss_calc(fluid_vel, length, hyd_diameter, corection_coeff, fluid_density=fluid_density):
    delta_p = corection_coeff*reynolds_ff(fluid_vel, hyd_diameter)*length*fluid_density*fluid_vel**2 / (hyd_diameter*2)
    return delta_p

###sweep iteration
def perform_flow_calc(piston_vel, needle_1_dis, needle_2_dis, 
                      inner_twin_BS_correction_coeff, outer_twin_correction_coeff, inner_twin_HS_correction_coeff, 
                      needle_1_dis_coeff, needle_2_dis_coeff, inner_twin_BS_loss_coeff, outer_twin_BS_loss_coeff, head_hole_loss_coeff, 
                      bore_area=bore_area, shaft_area=shaft_area,  outer_twin_area=outer_twin_area, outer_twin_hydraulic_diameter=outer_twin_hydraulic_diameter,
                      bore_ID=bore_ID, inner_twin_length=inner_twin_length, piston_head_length=piston_head_length, piston_displacement=piston_displacement,
                      inner_twin_BS_joint_approx_ID=inner_twin_BS_joint_approx_ID, outer_twin_BS_joint_approx_ID=outer_twin_BS_joint_approx_ID,
                      outer_twin_length=outer_twin_length, head_hole_flow_area=head_hole_flow_area, inner_twin_HS_hydraulic_diameter=inner_twin_HS_hydraulic_diameter, ext_reservoir_preload=ext_reservoir_preload, 
                      ):
    
    needle_1_area = needle_area(needle_1_dis)
    needle_2_area = needle_area(needle_2_dis)
    base_flow = piston_vel*bore_area
    head_flow = piston_vel*(bore_area-shaft_area)


    comp_inner_twin_BS_p = major_loss_calc(piston_vel, (inner_twin_length-piston_head_length), bore_ID, inner_twin_BS_correction_coeff[0])
    comp_inner_twin_BS_joint_p = joint_orifice_calc(base_flow, (np.pi*inner_twin_BS_joint_approx_ID**2/4), inner_twin_BS_loss_coeff[0])
    shim_1_delta_p = newton_method_shim_needle_calc(base_flow, needle_1_area, needle_1_dis_coeff[0])
    exh_2_delta_p = direct_solve_exhaust_needle_calc(head_flow, needle_2_area, needle_2_dis_coeff[0])
    comp_outer_twin_BS_joint_p = joint_orifice_calc(base_flow, (np.pi*outer_twin_BS_joint_approx_ID**2/4), outer_twin_BS_loss_coeff[0])
    comp_outer_twin_p = major_loss_calc((head_flow/outer_twin_area), outer_twin_length, outer_twin_hydraulic_diameter, outer_twin_correction_coeff[0])
    comp_head_hole_p = joint_orifice_calc(head_flow, head_hole_flow_area, head_hole_loss_coeff[0])
    comp_inner_twin_HS_p = major_loss_calc(piston_vel, piston_displacement, inner_twin_HS_hydraulic_diameter, inner_twin_HS_correction_coeff[0])

    reb_inner_twin_BS_p = major_loss_calc(piston_vel, (inner_twin_length-piston_head_length), bore_ID, inner_twin_BS_correction_coeff[1])
    reb_inner_twin_BS_joint_p = joint_orifice_calc(base_flow, (np.pi*inner_twin_BS_joint_approx_ID**2/4), inner_twin_BS_loss_coeff[1])
    shim_2_delta_p = newton_method_shim_needle_calc(head_flow, needle_2_area, needle_2_dis_coeff[1])
    exh_1_delta_p = direct_solve_exhaust_needle_calc(base_flow, needle_1_area, needle_1_dis_coeff[1])
    reb_outer_twin_BS_joint_p = joint_orifice_calc(base_flow, (np.pi*outer_twin_BS_joint_approx_ID**2/4), outer_twin_BS_loss_coeff[1])
    reb_outer_twin_p = major_loss_calc((head_flow/outer_twin_area), outer_twin_length, outer_twin_hydraulic_diameter, outer_twin_correction_coeff[1])
    reb_head_hole_p = joint_orifice_calc(head_flow, head_hole_flow_area, head_hole_loss_coeff[1])
    reb_inner_twin_HS_p = major_loss_calc(piston_vel, piston_displacement, inner_twin_HS_hydraulic_diameter, inner_twin_HS_correction_coeff[1])
    
    shock_pressure_drop = np.array([
        [comp_inner_twin_BS_p, comp_inner_twin_BS_joint_p, shim_1_delta_p, exh_2_delta_p, comp_outer_twin_BS_joint_p, comp_outer_twin_p, comp_head_hole_p, comp_inner_twin_HS_p],
        [reb_inner_twin_BS_p, reb_inner_twin_BS_joint_p, exh_1_delta_p, shim_2_delta_p, reb_outer_twin_BS_joint_p, reb_outer_twin_p, reb_head_hole_p, reb_inner_twin_HS_p]]
        )

    # compression_delta_p = np.sum(shock_pressure_drop[0,:])
    # rebound_delta_p = np.sum(shock_pressure_drop[1,:])
    compression_base_p = ext_reservoir_preload + np.sum(shock_pressure_drop[0,0:3])
    compression_head_p = ext_reservoir_preload - np.sum(shock_pressure_drop[0,3:])
    rebound_base_p = ext_reservoir_preload - np.sum(shock_pressure_drop[1,0:3])
    rebound_head_p = ext_reservoir_preload + np.sum(shock_pressure_drop[1,3:])
    compression_force = compression_base_p*bore_area-compression_head_p*(bore_area-shaft_area)
    rebound_force = rebound_base_p*bore_area-rebound_head_p*(bore_area-shaft_area)
    compression_damping_force = compression_force-ext_reservoir_preload*shaft_area
    rebound_damping_force = rebound_force-ext_reservoir_preload*shaft_area

    return compression_damping_force, rebound_damping_force




head_hole_data_CFD = [(1,5), (3,9), (6,15)]


exh_dis_coeff = 0.7  ## Optimize
shim_dis_coeff = 0.7  ## Optimize
needle_1_dis_coeff = [0.7, 0.7]  ## Optimize # [comp, reb]
needle_2_dis_coeff = [0.7, 0.7]  ## Optimize # [comp, reb]
inner_twin_BS_loss_coeff = [0.7, 0.7]  ## Optimize # [comp, reb]
outer_twin_BS_loss_coeff = [0.7, 0.7]  ## Optimize # [comp, reb]
head_hole_loss_coeff = [0.7, 0.7]  ## Optimize # [comp, reb]
inner_twin_BS_correction_coeff = [0.7, 0.7]  ## Optimize # [comp, reb]
inner_twin_HS_correction_coeff = [0.7, 0.7]  ## Optimize # [comp, reb]
outer_twin_correction_coeff = [0.7, 0.7]  ## Optimize # [comp, reb]


piston_vel_range = np.arange(0.5, 30.5, 0.5)
needle_num_pos = 3
needle_1_dis_range = np.linspace(0, d, needle_num_pos)
needle_2_dis_range = np.linspace(0, d, needle_num_pos)

sweep = np.zeros((2, len(piston_vel_range), needle_num_pos, needle_num_pos))
print(sweep)
print(sweep.shape)


for i, vel in enumerate(piston_vel_range):
    for j, n1 in enumerate(needle_1_dis_range):
        for k, n2 in enumerate(needle_2_dis_range):
            cdf, rdf = perform_flow_calc(vel, n1, n2, 
                                        inner_twin_BS_correction_coeff, outer_twin_correction_coeff, inner_twin_HS_correction_coeff, 
                                        needle_1_dis_coeff, needle_2_dis_coeff, inner_twin_BS_loss_coeff, outer_twin_BS_loss_coeff, head_hole_loss_coeff)
            sweep[0, i, j, k] = cdf
            sweep[1, i, j, k] = rdf


fig, axs = plt.subplots(2,needle_num_pos, figsize=(12,8))

for cr in range(2):
    for p, n2 in enumerate(needle_2_dis_range):
        for i, n1 in enumerate(needle_1_dis_range):
            y = sweep[cr,:,i,p]
            lab = f"needle 1 = {n1}"
            axs[cr,p].plot(piston_vel_range,y, label=lab)
            axs[cr,p].set_ylabel('damping force')
            axs[cr,p].set_xlabel('piston velocity')
            #axs[0,0].legend()

plt.show()
