# overwrite=1

# # echo "CREATING ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
# if [ ! -f ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh ] || [ "${overwrite}" = 1 ]; then
#     echo "#!/bin/sh" > ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
#     if [ "${overwrite}" = 1 ]; then
#         echo "OVERWROTE ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
#     else
#         echo "CREATED ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
#     fi
#     write_activate=1
# else
#     echo "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh   ALREADY EXISTS"
#     write_activate=0
# fi

# # echo "CREATING ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh"
# if [ ! -f ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh ] || [ "${overwrite}" = 1 ]; then
#     echo "#!/bin/sh" > ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh
#     if [ "${overwrite}" = 1 ]; then
#         echo "OVERWROTE ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh"
#     else
#         echo "CREATED ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh"
#     fi
#     write_deactivate=1
# else
#     echo "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh   ALREADY EXISTS"
#     write_deactivate=0
# fi

# if [ "${write_activate}" = 1 ]; then
#     echo "export EDF_ENV_DIR=\"${PWD}\"" >> ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
#     echo "source \${EDF_ENV_DIR}/catkin_ws/devel/setup.bash" >> ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
# else
#     echo "PASS ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
# fi

# if [ "${write_deactivate}" = 1 ]; then
#     echo "unset EDF_ENV_DIR" >> ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh
# else
#     echo "PASS ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh"
# fi

echo "Setting symbolic link from '${PWD}/devel/setup.sh' to '${PWD}/etc/conda/activate.d/ros_edf_interface_setup.sh'"
ln -sf ${PWD}/devel/setup.sh ${CONDA_PREFIX}/etc/conda/activate.d/ros_edf_interface_setup.sh
