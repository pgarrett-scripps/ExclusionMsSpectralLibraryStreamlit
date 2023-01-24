import streamlit as st
from exclusionms.components import DynamicExclusionTolerance
def get_tolerance():
    use_exact_charge = st.checkbox('Use exact charge', value=False)
    mass_tolerance = st.text_input(label='mass Tolerance', value='50')
    rt_tolerance = st.text_input(label='rt Tolerance', value='100')
    ook0_tolerance = st.text_input(label='ook0 Tolerance', value='0.05')
    intensity_tolerance = st.text_input(label='Intensity Tolerance', value='0.5')

    return DynamicExclusionTolerance(charge=use_exact_charge,
                                     mass=float(mass_tolerance) if mass_tolerance else None,
                                     rt=float(rt_tolerance) if rt_tolerance else None,
                                     ook0=float(ook0_tolerance) if ook0_tolerance else None,
                                     intensity=float(intensity_tolerance) if intensity_tolerance else None)