#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "grid_map_ipp::grid_map_ipp" for configuration ""
set_property(TARGET grid_map_ipp::grid_map_ipp APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(grid_map_ipp::grid_map_ipp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "/opt/ros/melodic/lib/libgrid_map_core.so"
  IMPORTED_LOCATION_NOCONFIG "/usr/local/libgrid_map_ipp.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS grid_map_ipp::grid_map_ipp )
list(APPEND _IMPORT_CHECK_FILES_FOR_grid_map_ipp::grid_map_ipp "/usr/local/libgrid_map_ipp.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
