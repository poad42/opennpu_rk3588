#!/usr/bin/env python3
# Generates a PJRT C plugin from pjrt_c_api.h: traced stubs for all fields +
# real overrides for the opennpu lifecycle/exec functions.
import re, subprocess

out = subprocess.check_output(
    ['awk', '/typedef struct PJRT_Api {/{f=1} f&&/^} PJRT_Api;/{f=0} f', 'pjrt_c_api.h']).decode()
fields = []
seen = set()
for m in re.finditer(r'_PJRT_API_STRUCT_FIELD\(\s*([A-Za-z_0-9]+)\s*\)', out):
    f = m.group(1)
    if f not in seen:
        seen.add(f); fields.append(f)

OVERRIDES = {
 'PJRT_Client_Create':'fn_Client_Create','PJRT_Client_Destroy':'fn_Client_Destroy',
 'PJRT_Client_PlatformName':'fn_Client_PlatformName','PJRT_Client_Devices':'fn_Client_Devices',
 'PJRT_Client_AddressableDevices':'fn_Client_AddressableDevices',
 'PJRT_Client_AddressableMemories':'fn_Client_AddressableMemories',
 'PJRT_Device_GetAttributes':'fn_Device_GetAttributes','PJRT_Error_Destroy':'fn_Error_Destroy',
 'PJRT_Error_Message':'fn_Error_Message','PJRT_Error_GetCode':'fn_Error_GetCode',
 'PJRT_Plugin_Initialize':'fn_Plugin_Initialize','PJRT_Client_BufferFromHostBuffer':'fn_Client_BufferFromHostBuffer',
 'PJRT_Buffer_ToHostBuffer':'fn_Buffer_ToHostBuffer','PJRT_Buffer_Destroy':'fn_Buffer_Destroy',
 'PJRT_Buffer_Delete':'fn_Buffer_Delete','PJRT_Buffer_ElementType':'fn_Buffer_ElementType',
 'PJRT_Buffer_Dimensions':'fn_Buffer_Dimensions','PJRT_Buffer_IsOnCpu':'fn_Buffer_IsOnCpu',
 'PJRT_Buffer_Device':'fn_Buffer_Device',
 'PJRT_Buffer_GetMemoryLayout':'fn_Buffer_GetMemoryLayout',
 'PJRT_Buffer_UnpaddedDimensions':'fn_Buffer_UnpaddedDimensions',
 'PJRT_Buffer_DynamicDimensionIndices':'fn_Buffer_DynamicDimensionIndices',
 'PJRT_Buffer_IsDeleted':'fn_Buffer_IsDeleted',
 'PJRT_Buffer_UnsafePointer':'fn_Buffer_UnsafePointer',
 'PJRT_Buffer_OnDeviceSizeInBytes':'fn_Buffer_OnDeviceSizeInBytes','PJRT_Buffer_Memory':'fn_Buffer_Memory',
 'PJRT_Buffer_ReadyEvent':'fn_Buffer_ReadyEvent','PJRT_Event_Destroy':'fn_Event_Destroy',
 'PJRT_Event_IsReady':'fn_Event_IsReady','PJRT_Event_Await':'fn_Event_Await',
 'PJRT_Event_Error':'fn_Event_Error','PJRT_Event_OnReady':'fn_Event_OnReady',
 'PJRT_Client_Compile':'fn_Client_Compile',
 'PJRT_Executable_NumReplicas':'fn_Executable_NumReplicas',
 'PJRT_Executable_NumPartitions':'fn_Executable_NumPartitions',
 'PJRT_Executable_NumOutputs':'fn_Executable_NumOutputs',
 'PJRT_Executable_OutputElementTypes':'fn_Executable_OutputElementTypes',
 'PJRT_Executable_OutputDimensions':'fn_Executable_OutputDimensions',
 'PJRT_Executable_OutputMemoryKinds':'fn_Executable_OutputMemoryKinds',
 'PJRT_Executable_Name':'fn_Executable_Name',
 'PJRT_Executable_SizeOfGeneratedCodeInBytes':'fn_Executable_SizeOfGeneratedCodeInBytes',
 'PJRT_Executable_GetCostAnalysis':'fn_Executable_GetCostAnalysis',
 'PJRT_Executable_OptimizedProgram':'fn_Executable_OptimizedProgram',
 'PJRT_LoadedExecutable_Execute':'fn_LoadedExecutable_Execute',
 'PJRT_LoadedExecutable_Destroy':'fn_LoadedExecutable_Destroy',
 'PJRT_LoadedExecutable_GetExecutable':'fn_LoadedExecutable_GetExecutable',
 'PJRT_LoadedExecutable_AddressableDevices':'fn_LoadedExecutable_AddressableDevices',
 'PJRT_LoadedExecutable_GetDeviceAssignment':'fn_LoadedExecutable_GetDeviceAssignment',
 'PJRT_LoadedExecutable_AddressableDeviceLogicalIds':'fn_LoadedExecutable_AddressableDeviceLogicalIds',
 'PJRT_Client_TopologyDescription':'fn_Client_TopologyDescription',
 'PJRT_TopologyDescription_PlatformName':'fn_TopologyDescription_PlatformName',
 'PJRT_TopologyDescription_PlatformVersion':'fn_TopologyDescription_PlatformVersion',
 'PJRT_TopologyDescription_Attributes':'fn_TopologyDescription_Attributes',
 'PJRT_TopologyDescription_Destroy':'fn_TopologyDescription_Destroy',
 'PJRT_Client_PlatformVersion':'fn_Client_PlatformVersion','PJRT_Plugin_Attributes':'fn_Plugin_Attributes',
 'PJRT_Client_ProcessIndex':'fn_Client_ProcessIndex','PJRT_Device_GetDescription':'fn_Device_GetDescription',
 'PJRT_DeviceDescription_Id':'fn_DeviceDescription_Id',
 'PJRT_DeviceDescription_ProcessIndex':'fn_DeviceDescription_ProcessIndex',
 'PJRT_DeviceDescription_Attributes':'fn_DeviceDescription_Attributes',
 'PJRT_DeviceDescription_Kind':'fn_DeviceDescription_Kind',
 'PJRT_DeviceDescription_DebugString':'fn_DeviceDescription_DebugString',
 'PJRT_DeviceDescription_ToString':'fn_DeviceDescription_ToString',
 'PJRT_Device_AddressableMemories':'fn_Device_AddressableMemories',
 'PJRT_Memory_AddressableByDevices':'fn_Memory_AddressableByDevices',
 'PJRT_Client_DefaultDeviceAssignment':'fn_Client_DefaultDeviceAssignment',
 'PJRT_Client_LookupDevice':'fn_Client_LookupDevice',
 'PJRT_Device_IsAddressable':'fn_Device_IsAddressable',
 'PJRT_Device_LocalHardwareId':'fn_Device_LocalHardwareId',
 'PJRT_Device_DefaultMemory':'fn_Device_DefaultMemory',
 'PJRT_Device_MemoryStats':'fn_Device_MemoryStats',
 'PJRT_Memory_Id':'fn_Memory_Id',
 'PJRT_Memory_Kind':'fn_Memory_Kind',
 'PJRT_Memory_Kind_Id':'fn_Memory_Kind_Id',
 'PJRT_Memory_DebugString':'fn_Memory_DebugString',
 'PJRT_Memory_ToString':'fn_Memory_ToString',
 'PJRT_TopologyDescription_GetDeviceDescriptions':'fn_TopologyDescription_GetDeviceDescriptions',
 'PJRT_TopologyDescription_Fingerprint':'fn_TopologyDescription_Fingerprint',
 'PJRT_TopologyDescription_GetMemorySpaceKindIds':'fn_TopologyDescription_GetMemorySpaceKindIds',
}

decls = '\n'.join(f'static PJRT_Error* {v}(void* a);' for v in sorted(set(OVERRIDES.values())))
stubs = ''

assigns = ''
for f in fields:
    impl = OVERRIDES.get(f)
    src = impl if impl else 'pjrt_stub_null'
    assigns += f'    api->{f} = (typeof(api->{f})){src};\n'

body = f'''/* Auto-generated PJRT C plugin for the RK3588 NPU (opennpu, open stack).
 * GetPjrtApi() returns PJRT_Api (v0.112 (exact match for jaxlib 0.10.2 build))
 * with all {len(fields)} function pointers: real opennpu impls + traced stubs.
 */
#include "pjrt_c_api.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

static PJRT_Error* make_err(const char* m);
static PJRT_Error* pjrt_stub_null(void* a){{ (void)a; return NULL; }}
{decls}

{stubs}

#include "pjrt_npu_impl.c"

const PJRT_Api* GetPjrtApi(void) {{
  static union {{ PJRT_Api api; char pad[8192]; }} u;
  PJRT_Api* api = &u.api;
  static int init = 0;
  if (!init) {{
    memset(&u, 0, sizeof(u));
    api->struct_size = sizeof(PJRT_Api);
    api->pjrt_api_version.struct_size = sizeof(PJRT_Api_Version);
    api->pjrt_api_version.major_version = PJRT_API_MAJOR;
    api->pjrt_api_version.minor_version = PJRT_API_MINOR;
{assigns}    init = 1;
  }}
  return api;
}}
'''
open('pjrt_npu.c', 'w').write(body)
print(f'wrote pjrt_npu.c ({len(fields)} fields, {len(OVERRIDES)} overrides, {len(body)} bytes)')