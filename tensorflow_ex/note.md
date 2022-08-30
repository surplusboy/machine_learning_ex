# 자잘한 문제 해결 방안

1. tensor flow 내의 kears 패키지의 자동완성이 되지 않는 문제 해결 방법

site-packages/tensor-flow/&#95;&#95;init&#95;&#95;.py 내의 파일을 수정하면 자동완성이 되는것을 볼 수 있다

#### 변경 전 ####
```before
_keras_module = "keras.api._v2.keras"
keras = _LazyLoader("keras", globals(), _keras_module)
_module_dir = _module_util.get_parent_dir_for_name(_keras_module)
if _module_dir:
  _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, "keras", keras)
```


#### 변경 후 ####
```after
import typing as _typing
if _typing.TYPE_CHECKING:
  from keras.api._v2 import keras
else:
  _keras_module = "keras.api._v2.keras"
  keras = _LazyLoader("keras", globals(), _keras_module)
  _module_dir = _module_util.get_parent_dir_for_name(_keras_module)
  if _module_dir:
    _current_module.__path__ = [_module_dir] + _current_module.__path__
  setattr(_current_module, "keras", keras)
```