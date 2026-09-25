"""Namespace generated model symbols and emit the public header/ABI metadata."""
import hashlib
import json
from pathlib import Path
import re


def validate_prefix(prefix):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', prefix):
        raise ValueError('symbol prefix must start with a letter and contain only ASCII letters, digits or underscores')
    # The prefix also names the opaque model type, so a bare C keyword is not
    # usable even though the suffixed function names would be identifiers.
    if prefix in {'auto', 'break', 'case', 'char', 'const', 'continue', 'default',
                  'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto',
                  'if', 'inline', 'int', 'long', 'register', 'restrict', 'return',
                  'short', 'signed', 'sizeof', 'static', 'struct', 'switch',
                  'typedef', 'union', 'unsigned', 'void', 'volatile', 'while'}:
        raise ValueError('symbol prefix must not be a C keyword')
    return prefix


def api_code(manifest):
    """Append typed adapters; the implementation/experimental ABI stays intact."""
    return '''
static int native_preset_supported(uint32_t preset) {
    switch (preset) {
    case DPDF_PRESET_FP32: return 1;
    case DPDF_PRESET_INT8_SELECTIVE: return dpdf_has_avx2();
    default: return 0;
    }
}
static dpdf_native_model *native_create(const float *w,size_t count,uint32_t preset) {
    if (!native_preset_supported(preset)) return NULL;
    if (preset==DPDF_PRESET_INT8_SELECTIVE)
        return (dpdf_native_model *)dpdf_model_create_config(w,count,DPDF_EXPERIMENTAL_INT8,8,7);
    return (dpdf_native_model *)dpdf_model_create_config(w,count,DPDF_AUTO,0,0);
}
static int native_process(dpdf_native_model *m,const float *x,const float *s,float *y,float *t) {
    return dpdf_model_process((dpdf_model *)m,x,s,y,t);
}
static void native_destroy(dpdf_native_model *m) {
    dpdf_model_destroy((dpdf_model *)m);
}
static size_t native_owned_bytes(const dpdf_native_model *m) {
    return dpdf_model_owned_bytes((const dpdf_model *)m);
}
const dpdf_native_api_v1 *dpdf_model_get_api(uint32_t version) {
    static const dpdf_native_api_v1 api={
        DPDF_NATIVE_ABI_VERSION, sizeof(dpdf_native_api_v1),
        "MODEL_NAME", "WEIGHTS_SHA", 48000, 480,
        SPECTRUM_SIZE, STATE_SIZE, WEIGHT_COUNT,
        native_preset_supported, native_create, dpdf_model_init_state,
        native_process, native_destroy, native_owned_bytes
    };
    return version==DPDF_NATIVE_ABI_VERSION ? &api : NULL;
}
'''.replace('MODEL_NAME', manifest['profile']).replace(
        'WEIGHTS_SHA', manifest['weights_sha256']).replace(
        'SPECTRUM_SIZE', str(manifest['spectrum_size'])).replace(
        'STATE_SIZE', str(manifest['state_size'])).replace(
        'WEIGHT_COUNT', str(manifest['weight_floats']))


def finalize(folder, symbol_prefix='dpdf_model', extended=False):
    validate_prefix(symbol_prefix)
    path = folder / 'generated_model.c'
    code = path.read_text()
    manifest = json.loads((folder / 'manifest.json').read_text())
    if extended:
        code += api_code(manifest)
    # Replace complete model identifiers, including the opaque struct tag.
    # Kernel symbols are intentionally shared and compiled exactly once.
    def rename(text):
        return re.sub(r'\bdpdf_model(?:_[A-Za-z0-9_]+)?\b',
                      lambda m: symbol_prefix + m[0][len('dpdf_model'):], text)
    header = (Path(__file__).parent / 'full_model.h').read_text()
    header = header.replace('DPDF_FULL_MODEL_H', f'DPDF_GENERATED_{symbol_prefix}_H')
    header = rename(header)
    code = rename(code.replace('#include "full_model.h"', '#include "generated_model.h"'))
    path.write_text(code, encoding='utf-8', newline='\n')
    (folder / 'generated_model.h').write_text(header, encoding='utf-8', newline='\n')
    manifest['symbol_prefix'] = symbol_prefix
    manifest['native_abi_version'] = 1 if extended else None
    manifest['artifacts'] = {
        name: {'sha256': hashlib.sha256((folder / name).read_bytes()).hexdigest(),
               'bytes': (folder / name).stat().st_size}
        for name in ('generated_model.c', 'generated_model.h', 'weights.f32')
    }
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n',
                                         encoding='utf-8', newline='\n')
