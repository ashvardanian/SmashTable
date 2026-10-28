# Install rules and the CMake package files.

option(SMASHTABLE_INSTALL "Install headers and CMake package files" ${PROJECT_IS_TOP_LEVEL})

if (SMASHTABLE_INSTALL)
    # Installation
    install(
        DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.hpp"
    )

    install(
        TARGETS smashtable_header
        EXPORT smashtableTargets
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(
        EXPORT smashtableTargets
        FILE smashtableTargets.cmake
        NAMESPACE smashtable::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smashtable
    )

    # Package config for find_package(smashtable)
    include(CMakePackageConfigHelpers)

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/smashtableConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/smashtableConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/smashtableConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smashtable
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
    )

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/smashtableConfig.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/smashtableConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smashtable
    )
endif ()
