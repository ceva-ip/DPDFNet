"""Strict compiler experiments for the preserved Oct7 exact profile.

The numerical C and assembly sources are untouched. PGO requires a two-phase
build in the same target directory; see README.md and train.py.
"""
import hashlib
from pathlib import Path


VARIANTS = ('compiler_ipo_nointerpose', 'compiler_pgo',
            'compiler_pgo_ipo_nointerpose', 'compiler_hot_align')
PGO_VARIANTS = ('compiler_pgo', 'compiler_pgo_ipo_nointerpose')
BASE_CMAKE_SHA256 = 'd2f40b6fc8b198d8e587eb5f7d49d76a5f2de2370f9c1a8a2d1a450e6e0f532b'
MARKER = '# Oct7b compiler experiment: numerical operations remain unchanged.'


def apply(name, source):
    if name not in VARIANTS:
        raise ValueError(name)
    path = Path(source)/'CMakeLists.txt'
    code = path.read_text()
    if MARKER in code:
        raise RuntimeError('Only one compiler profile may be selected per source snapshot')
    if hashlib.sha256(code.encode()).hexdigest() != BASE_CMAKE_SHA256:
        raise RuntimeError('Compiler experiment requires the preserved Oct7 CMake baseline')
    code += '\n'+MARKER+'\n'
    code += '''if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "The Oct7b compiler profile is a GCC-only experiment")
endif()
'''
    if 'ipo_nointerpose' in name:
        code += '''include(CheckIPOSupported)
check_ipo_supported(RESULT dpdf_oct7b_ipo_supported OUTPUT dpdf_oct7b_ipo_error LANGUAGES C)
if(NOT dpdf_oct7b_ipo_supported)
  message(FATAL_ERROR "IPO unsupported: ${dpdf_oct7b_ipo_error}")
endif()
foreach(dpdf_oct7b_target dpdf_kernels dpdf_dprnn dpdf_full)
  if(TARGET ${dpdf_oct7b_target})
    set_property(TARGET ${dpdf_oct7b_target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    target_compile_options(${dpdf_oct7b_target} PRIVATE -fno-semantic-interposition)
  endif()
endforeach()
'''
    if name in PGO_VARIANTS:
        code += '''set(DPDF_OCT7B_PGO_MODE "OFF" CACHE STRING "Oct7b PGO: OFF, GENERATE or USE")
set(DPDF_OCT7B_PROFILE_DIR "" CACHE PATH "GCC profiles tied to this exact target directory")
set(dpdf_oct7b_profile_flag "")
if(NOT DPDF_OCT7B_PGO_MODE STREQUAL "OFF" AND
   (NOT DPDF_OCT7B_PROFILE_DIR OR NOT IS_ABSOLUTE "${DPDF_OCT7B_PROFILE_DIR}"))
  message(FATAL_ERROR "An absolute DPDF_OCT7B_PROFILE_DIR is required")
endif()
if(DPDF_OCT7B_PGO_MODE STREQUAL "GENERATE")
  set(dpdf_oct7b_profile_flag "-fprofile-generate=${DPDF_OCT7B_PROFILE_DIR}")
elseif(DPDF_OCT7B_PGO_MODE STREQUAL "USE")
  file(GLOB_RECURSE dpdf_oct7b_profile_files "${DPDF_OCT7B_PROFILE_DIR}/*.gcda")
  if(NOT dpdf_oct7b_profile_files)
    message(FATAL_ERROR "PGO USE requires flushed training profiles")
  endif()
  set(dpdf_oct7b_profile_flag "-fprofile-use=${DPDF_OCT7B_PROFILE_DIR}")
elseif(NOT DPDF_OCT7B_PGO_MODE STREQUAL "OFF")
  message(FATAL_ERROR "Select GENERATE, train, then USE in the same build directory")
endif()
if(dpdf_oct7b_profile_flag)
foreach(dpdf_oct7b_target dpdf_kernels dpdf_dprnn dpdf_full)
  if(TARGET ${dpdf_oct7b_target})
    target_compile_options(${dpdf_oct7b_target} PRIVATE
      "$<$<COMPILE_LANGUAGE:C>:${dpdf_oct7b_profile_flag}>")
    target_link_options(${dpdf_oct7b_target} PRIVATE "${dpdf_oct7b_profile_flag}")
  endif()
endforeach()
endif()
'''
    if name == 'compiler_hot_align':
        # Tune only hot dispatched files; leave graph dispatch/cold constructors
        # at their default layout. No global -march or arithmetic option changes.
        code += '''set_property(SOURCE avx2.c int8.c APPEND PROPERTY COMPILE_OPTIONS
  "-falign-functions=32" "-falign-loops=32:16" "-falign-jumps=16:8")
'''
    path.write_text(code)


def profile_dir(target):
    target = Path(target).resolve()
    return target.parent/(target.name+'_profiles')


def build_options(name, source, target, phase='use'):
    """Extra CMake arguments, with a unique path for each target's counters."""
    del source  # Explicit interface for generic experiment drivers.
    if name not in VARIANTS:
        raise ValueError(name)
    if name not in PGO_VARIANTS:
        return []
    if phase not in ('generate', 'use', 'plain'):
        raise ValueError(phase)
    mode = 'OFF' if phase=='plain' else phase.upper()
    return [f'-DDPDF_OCT7B_PGO_MODE={mode}',
            f'-DDPDF_OCT7B_PROFILE_DIR={profile_dir(target)}']
