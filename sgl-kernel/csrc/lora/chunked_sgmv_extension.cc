#include <torch/extension.h>
#include "chunked_sgmv_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunked_sgmv_lora_shrink", &chunked_sgmv_lora_shrink,
          "Chunked SGMV LoRA shrink operation (CUDA)",
          py::arg("x"),
          py::arg("weights"),
          py::arg("seg_indptr"),
          py::arg("weight_indices"),
          py::arg("lora_ranks"),
          py::arg("permutation"),
          py::arg("num_segments"),
          py::arg("num_slices") = 1
    );

    m.def("chunked_sgmv_lora_expand", &chunked_sgmv_lora_expand,
          "Chunked SGMV LoRA expand operation (CUDA)",
          py::arg("x"),
          py::arg("lora_weight_b"),
          py::arg("seg_indptr"),
          py::arg("weight_indices"),
          py::arg("lora_ranks"),
          py::arg("permutation"),
          py::arg("scalings"),
          py::arg("slice_offsets"),
          py::arg("num_segments"),
          py::arg("max_slice_size"),
          py::arg("base_output") = py::none()
    );
}