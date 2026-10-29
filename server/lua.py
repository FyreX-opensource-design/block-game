import ctypes


_LIB = ctypes.CDLL("liblua.so.5.5")
_STATE = ctypes.c_void_p
_LIB.luaL_newstate.restype = _STATE
_LIB.luaL_openselectedlibs.argtypes = [_STATE, ctypes.c_int, ctypes.c_int]
_LIB.luaL_loadfilex.argtypes = [_STATE, ctypes.c_char_p, ctypes.c_char_p]
_LIB.luaL_loadfilex.restype = ctypes.c_int
_LIB.lua_pcallk.argtypes = [_STATE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ssize_t, ctypes.c_void_p]
_LIB.lua_pcallk.restype = ctypes.c_int
_LIB.lua_type.argtypes = [_STATE, ctypes.c_int]
_LIB.lua_type.restype = ctypes.c_int
_LIB.lua_tonumberx.argtypes = [_STATE, ctypes.c_int, ctypes.c_void_p]
_LIB.lua_tonumberx.restype = ctypes.c_double
_LIB.lua_tolstring.argtypes = [_STATE, ctypes.c_int, ctypes.c_void_p]
_LIB.lua_tolstring.restype = ctypes.c_char_p
_LIB.lua_toboolean.argtypes = [_STATE, ctypes.c_int]
_LIB.lua_toboolean.restype = ctypes.c_int
_LIB.lua_next.argtypes = [_STATE, ctypes.c_int]
_LIB.lua_next.restype = ctypes.c_int
_LIB.lua_pushnil.argtypes = [_STATE]
_LIB.lua_settop.argtypes = [_STATE, ctypes.c_int]
_LIB.lua_close.argtypes = [_STATE]

_BOOLEAN = 1
_NUMBER = 3
_STRING = 4
_TABLE = 5


def _read_table(state, path):
    _LIB.lua_pushnil(state)
    pairs = []
    while _LIB.lua_next(state, -2):
        key_type = _LIB.lua_type(state, -2)
        if key_type == _STRING:
            raw = _LIB.lua_tolstring(state, -2, None)
            key = raw.decode() if raw is not None else ""
            numeric = False
        elif key_type == _NUMBER:
            key = float(_LIB.lua_tonumberx(state, -2, None))
            numeric = True
        else:
            raise ValueError(f"{path}: table keys must be strings or numbers")
        pairs.append((numeric, key, _read_value(state, path, key)))
        _LIB.lua_settop(state, -2)
    if any(numeric for numeric, _key, _value in pairs) and any(not numeric for numeric, _key, _value in pairs):
        raise ValueError(f"{path}: a table cannot mix a list with named fields")
    if pairs and pairs[0][0]:
        ordered = sorted(pairs, key=lambda item: item[1])
        indexes = []
        for _numeric, key, _value in ordered:
            if int(key) != key:
                raise ValueError(f"{path}: list indexes must be whole numbers")
            indexes.append(int(key))
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError(f"{path}: lists must be packed from 1")
        return [value for _numeric, _key, value in ordered]
    return {key: value for _numeric, key, value in pairs}


def _read_value(state, path, key):
    kind = _LIB.lua_type(state, -1)
    if kind == _STRING:
        text = _LIB.lua_tolstring(state, -1, None)
        return text.decode() if text is not None else ""
    if kind == _NUMBER:
        return float(_LIB.lua_tonumberx(state, -1, None))
    if kind == _BOOLEAN:
        return bool(_LIB.lua_toboolean(state, -1))
    if kind == _TABLE:
        return _read_table(state, path)
    raise ValueError(f"{path}: {key} must be a string, number, boolean, or table")


def load_table(path):
    state = _LIB.luaL_newstate()
    if not state:
        raise RuntimeError("could not start Lua")
    try:
        _LIB.luaL_openselectedlibs(state, -1, 0)
        if _LIB.luaL_loadfilex(state, str(path).encode(), None) or _LIB.lua_pcallk(state, 0, 1, 0, 0, None):
            message = _LIB.lua_tolstring(state, -1, None)
            detail = message.decode() if message else "Lua error"
            raise ValueError(f"{path}: {detail}")
        if _LIB.lua_type(state, -1) != _TABLE:
            raise ValueError(f"{path}: Lua file must return a table")
        return _read_table(state, path)
    finally:
        _LIB.lua_close(state)
