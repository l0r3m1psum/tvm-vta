/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file sim_driver.cc
 * \brief VTA driver for simulated backend.
 */
#include <vta/driver.h>
#include <vta/hw_spec.h>
#include <tvm/runtime/registry.h>
#include <vta/sim_tlpp.h>
#include <type_traits>
#include <mutex>
#include <map>
#include <unordered_map>
#include <cstring>
#include <sstream>

#include "../vmem/virtual_memory.h"

namespace vta {
namespace sim {

/*! \brief debug flag for skipping computation */
enum DebugFlagMask {
  kSkipExec = 1
};

/*!
 * \brief Helper class to pack and unpack bits
 *  Applies truncation when pack to low level bits.
 *
 * \tparam bits The number of bits in integer.
 * \note This implementation relies on little endian.
 */
template<uint32_t bits>
class BitPacker {
 public:
  explicit BitPacker(void* data) {
    data_ = static_cast<uint32_t*>(data);
  }

  uint32_t GetUnsigned(uint32_t index) const {
    if (bits == 32) {
      return data_[index];
    } else if (bits == 16) {
      return reinterpret_cast<uint16_t*>(data_)[index];
    } else if (bits == 8) {
      return reinterpret_cast<uint8_t*>(data_)[index];
    } else {
      uint32_t offset = index / kNumPackElem;
      uint32_t shift = index % kNumPackElem;
      return (data_[offset] >> shift) & kMask;
    }
  }

  int32_t GetSigned(uint32_t index) const {
    if (bits == 32) {
      return reinterpret_cast<int32_t*>(data_)[index];
    } else if (bits == 16) {
      return reinterpret_cast<int16_t*>(data_)[index];
    } else if (bits == 8) {
      return reinterpret_cast<int8_t*>(data_)[index];
    } else {
      uint32_t offset = index / kNumPackElem;
      uint32_t shift = (index % kNumPackElem) * bits;
      int32_t uvalue = static_cast<int32_t>(
          (data_[offset] >> shift) & kMask);
      int kleft = 32 - bits;
      return (uvalue << kleft) >> kleft;
    }
  }

  void SetUnsigned(uint32_t index, uint32_t value) {
    if (bits == 32) {
      data_[index] = value;
    } else if (bits == 16) {
      reinterpret_cast<uint16_t*>(data_)[index] = value;
    } else if (bits == 8) {
      reinterpret_cast<uint8_t*>(data_)[index] = value;
    } else {
      uint32_t offset = index / kNumPackElem;
      uint32_t shift = (index % kNumPackElem) * bits;
      data_[offset] &= (~(kMask << shift));
      data_[offset] |= (value & kMask) << shift;
    }
  }

  void SetSigned(uint32_t index, int32_t value) {
    if (bits == 32) {
      reinterpret_cast<int32_t*>(data_)[index] = value;
    } else if (bits == 16) {
      reinterpret_cast<int16_t*>(data_)[index] = value;
    } else if (bits == 8) {
      reinterpret_cast<int8_t*>(data_)[index] = value;
    } else {
      uint32_t offset = index / kNumPackElem;
      uint32_t shift = (index % kNumPackElem) * bits;
      data_[offset] &= (~(kMask << shift));
      data_[offset] |= static_cast<uint32_t>(value & kMask) << shift;
    }
  }

 private:
  uint32_t* data_;
  static constexpr uint32_t kNumPackElem = 32 / bits;
  static constexpr uint32_t kMask = (1U << (bits >= 32U ? 31U : bits)) - 1U;
};

/*!
 * \brief DRAM memory manager
 *  Implements simple paging to allow physical address translation.
 */
using DRAM = ::vta::vmem::VirtualMemoryManager;

/*!
 * \brief Register file.
 * \tparam kBits Number of bits of one value.
 * \tparam kLane Number of lanes in one element.
 * \tparam kMaxNumElem Maximum number of element.
 */
template<int kBits, int kLane, int kMaxNumElem>
class SRAM {
 public:
  /*! \brief Bytes of single vector element */
  static const int kElemBytes = (kBits * kLane + 7) / 8;
  /*! \brief content data type */
  using DType = typename std::aligned_storage<kElemBytes, kElemBytes>::type;
  SRAM() {
    data_ = new DType[kMaxNumElem];
  }
  ~SRAM() {
    delete [] data_;
  }
  // Get the i-th index
  void* BeginPtr(uint32_t index) {
    CHECK_LT(index, kMaxNumElem);
    return &(data_[index]);
  }
  // Execute the load instruction on this SRAM
  void Load(const VTAMemInsn* op,
            DRAM* dram,
            uint64_t* load_counter,
            bool skip_exec) {
    load_counter[0] += (op->x_size * op->y_size) * kElemBytes;
    if (skip_exec) return;
    DType* sram_ptr = data_ + op->sram_base;
    // The DRAM holds packed 4 bit elements that are extended to 8 bits, only
    // for the memories with 8 bit elements (input and weight)
    bool int4 = op->int4 && kBits == 8;
    int dram_elem_bytes = int4 ? kElemBytes / 2 : kElemBytes;
    uint8_t* dram_ptr = static_cast<uint8_t*>(dram->GetAddr(
        op->dram_base * dram_elem_bytes));
    uint64_t xtotal = op->x_size + op->x_pad_0 + op->x_pad_1;
    uint32_t ytotal = op->y_size + op->y_pad_0 + op->y_pad_1;
    uint64_t sram_end = op->sram_base + xtotal * ytotal;
    CHECK_LE(sram_end, kMaxNumElem);
    memset(sram_ptr, 0, kElemBytes * xtotal * op->y_pad_0);
    sram_ptr += xtotal * op->y_pad_0;

    for (uint32_t y = 0; y < op->y_size; ++y) {
      memset(sram_ptr, 0, kElemBytes * op->x_pad_0);
      sram_ptr += op->x_pad_0;
      if (int4) {
        // The low nibble of a byte is the first element
        uint8_t* sram_bytes = reinterpret_cast<uint8_t*>(sram_ptr);
        for (uint64_t i = 0; i < static_cast<uint64_t>(kElemBytes) * op->x_size; ++i) {
          uint8_t nibble = (dram_ptr[i / 2] >> (4 * (i % 2))) & 0xF;
          if (!op->is_unsigned && (nibble & 0x8)) {
            nibble |= 0xF0;
          }
          sram_bytes[i] = nibble;
        }
      } else {
        memcpy(sram_ptr, dram_ptr, kElemBytes * op->x_size);
      }
      sram_ptr += op->x_size;
      memset(sram_ptr, 0, kElemBytes * op->x_pad_1);
      sram_ptr += op->x_pad_1;
      dram_ptr += dram_elem_bytes * op->x_stride;
    }
    memset(sram_ptr, 0, kElemBytes * xtotal * op->y_pad_1);
  }

  // This is for load 8bits to ACC only, sign or zero extended
  void Load_int8(const VTAMemInsn* op,
            DRAM* dram,
            uint64_t* load_counter,
            bool skip_exec,
            bool is_unsigned) {
    CHECK_EQ(kBits, VTA_ACC_WIDTH);

    // TODO(zhanghao): extend to other width
    CHECK_EQ(VTA_ACC_WIDTH, 32);
    CHECK_EQ(VTA_INP_WIDTH, 8);

    int factor = VTA_ACC_WIDTH / VTA_INP_WIDTH;
    load_counter[0] += (op->x_size * op->y_size) * kElemBytes;
    if (skip_exec) return;
    DType* sram_ptr = data_ + op->sram_base;
    int8_t* dram_ptr = static_cast<int8_t*>(dram->GetAddr(
        op->dram_base * kElemBytes / factor));
    uint64_t xtotal = op->x_size + op->x_pad_0 + op->x_pad_1;
    uint32_t ytotal = op->y_size + op->y_pad_0 + op->y_pad_1;
    uint64_t sram_end = op->sram_base + xtotal * ytotal;
    CHECK_LE(sram_end, kMaxNumElem);
    memset(sram_ptr, 0, kElemBytes * xtotal * op->y_pad_0);
    sram_ptr += xtotal * op->y_pad_0;

    for (uint32_t y = 0; y < op->y_size; ++y) {
      memset(sram_ptr, 0, kElemBytes * op->x_pad_0);
      sram_ptr += op->x_pad_0;

      int32_t* sram_ele_ptr = (int32_t*)sram_ptr;
      for (uint32_t x = 0; x < op->x_size * VTA_BATCH * VTA_BLOCK_OUT; ++x) {
        if (is_unsigned) {
          *(sram_ele_ptr + x) = (int32_t)(uint8_t)*(dram_ptr + x);
        } else {
          *(sram_ele_ptr + x) = (int32_t)*(dram_ptr + x);
        }
      }
      sram_ptr += op->x_size;

      memset(sram_ptr, 0, kElemBytes * op->x_pad_1);
      sram_ptr += op->x_pad_1;

      // dram one element is 1 bytes rather than 4 bytes
      dram_ptr += kElemBytes / factor * op->x_stride;
    }
    memset(sram_ptr, 0, kElemBytes * xtotal * op->y_pad_1);
  }


  // Execute the store instruction on this SRAM apply trucation.
  // This relies on the elements is 32 bits
  template<int target_bits>
  void TruncStore(const VTAMemInsn* op, DRAM* dram) {
    CHECK_EQ(op->x_pad_0, 0);
    CHECK_EQ(op->x_pad_1, 0);
    CHECK_EQ(op->y_pad_0, 0);
    CHECK_EQ(op->y_pad_1, 0);
    int target_width = (target_bits * kLane + 7) / 8;
    BitPacker<kBits> src(data_ + op->sram_base);
    BitPacker<target_bits> dst(dram->GetAddr(op->dram_base * target_width));
    for (uint32_t y = 0; y < op->y_size; ++y) {
      for (uint32_t x = 0; x < op->x_size; ++x) {
        uint32_t sram_base = y * op->x_size + x;
        uint32_t dram_base = y * op->x_stride + x;
        for (int i = 0; i < kLane; ++i) {
          dst.SetSigned(dram_base * kLane + i,
                        src.GetSigned(sram_base * kLane +i));
        }
      }
    }
  }

 private:
  /*! \brief internal data content */
  DType* data_;
};


/*!
 * \brief Memory information of special memory region.
 *  Use MemoryInfo as its container type
 */
class Profiler {
 public:
  /*! \brief The memory load statistics */
  uint64_t inp_load_nbytes{0};
  /*! \brief The memory load statistics */
  uint64_t wgt_load_nbytes{0};
  /*! \brief The ACC memory load statistics */
  uint64_t acc_load_nbytes{0};
  /*! \brief The ACC memory load statistics */
  uint64_t uop_load_nbytes{0};
  /*! \brief The ACC memory load statistics */
  uint64_t out_store_nbytes{0};
  /*! \brief instr counter for gemm */
  uint64_t gemm_counter{0};
  /*! \brief instr counter for ALU ops */
  uint64_t alu_counter{0};
  /*! \brief set debug mode */
  int64_t debug_flag{0};
  /*! \brief clear the profiler */
  void Clear() {
    inp_load_nbytes = 0;
    wgt_load_nbytes = 0;
    acc_load_nbytes = 0;
    uop_load_nbytes = 0;
    out_store_nbytes = 0;
    gemm_counter = 0;
    alu_counter = 0;
  }
  /*! \return Whether we should skip execution. */
  bool SkipExec() const {
    return (debug_flag & DebugFlagMask::kSkipExec) != 0;
  }

  std::string AsJSON() {
    std::ostringstream os;
    os << "{\n"
       << " \"inp_load_nbytes\":" << inp_load_nbytes << ",\n"
       << " \"wgt_load_nbytes\":" << wgt_load_nbytes << ",\n"
       << " \"acc_load_nbytes\":" << acc_load_nbytes << ",\n"
       << " \"uop_load_nbytes\":" << uop_load_nbytes << ",\n"
       << " \"out_store_nbytes\":" << out_store_nbytes << ",\n"
       << " \"gemm_counter\":" << gemm_counter << ",\n"
       << " \"alu_counter\":" << alu_counter << "\n"
       <<"}\n";
    return os.str();
  }

  static Profiler* ThreadLocal() {
    static thread_local Profiler inst;
    return &inst;
  }
};


// Simulate device
// TODO(tqchen,thierry): queue based event driven simulation.
class Device {
 public:
  Device() {
    prof_ = Profiler::ThreadLocal();
    dram_ = DRAM::Global();
    ptlpp = TlppVerify::Global();
  }

  int Run(vta_phy_addr_t insn_phy_addr,
          uint32_t insn_count,
          uint32_t wait_cycles) {
    VTAGenericInsn* insn = static_cast<VTAGenericInsn*>(
        dram_->GetAddr(insn_phy_addr));
    finish_counter_ = 0;
    for (uint32_t i = 0; i < insn_count; ++i) {
      this->Run(insn + i);
    }
    this->TlppSynchronization();
    return 0;
  }

 private:
  static void Run_Insn(const VTAGenericInsn* insn, void * dev) {
    Device * device = reinterpret_cast<Device *> (dev);
    const VTAMemInsn* mem = reinterpret_cast<const VTAMemInsn*>(insn);
    const VTAGemInsn* gem = reinterpret_cast<const VTAGemInsn*>(insn);
    const VTAAluInsn* alu = reinterpret_cast<const VTAAluInsn*>(insn);
    switch (mem->opcode) {
      case VTA_OPCODE_LOAD: device->RunLoad(mem); break;
      case VTA_OPCODE_STORE: device->RunStore(mem); break;
      case VTA_OPCODE_GEMM: device->RunGEMM(gem, false); break;
      case VTA_OPCODE_GEMM_BINARY: device->RunGEMM(gem, true); break;
      case VTA_OPCODE_ALU: device->RunALU(alu); break;
      case VTA_OPCODE_FINISH: ++(device->finish_counter_); break;
      default: {
        LOG(FATAL) << "Unknown op_code" << mem->opcode;
      }
    }
  }

 private:
  void Run(const VTAGenericInsn* insn) {
    ptlpp->TlppPushInsn(insn);
  }

  void TlppSynchronization(void) {
    ptlpp->TlppSynchronization(Run_Insn, reinterpret_cast<void *> (this));
  }

  void RunLoad(const VTAMemInsn* op) {
    if (op->x_size == 0) return;
    if (op->memory_type == VTA_MEM_ID_INP) {
      inp_.Load(op, dram_, &(prof_->inp_load_nbytes), prof_->SkipExec());
    } else if (op->memory_type == VTA_MEM_ID_WGT) {
      wgt_.Load(op, dram_, &(prof_->wgt_load_nbytes), prof_->SkipExec());
    } else if (op->memory_type == VTA_MEM_ID_ACC) {
      acc_.Load(op, dram_, &(prof_->acc_load_nbytes), prof_->SkipExec());
    } else if (op->memory_type == VTA_MEM_ID_UOP) {
      // always load in uop, since uop is stateful
      // subsequent non-debug mode exec can depend on it.
      uop_.Load(op, dram_, &(prof_->uop_load_nbytes), false);
    } else if (op->memory_type == VTA_MEM_ID_ACC_8BIT) {
      acc_.Load_int8(op, dram_, &(prof_->acc_load_nbytes), prof_->SkipExec(),
                     op->is_unsigned);
    } else {
      LOG(FATAL) << "Unknown memory_type=" << op->memory_type;
    }
  }

  void RunStore(const VTAMemInsn* op) {
    if (op->x_size == 0) return;
    if (op->memory_type == VTA_MEM_ID_OUT) {
      prof_->out_store_nbytes += (
          op->x_size * op->y_size * VTA_BATCH * VTA_BLOCK_OUT * VTA_OUT_WIDTH / 8);
      if (!prof_->SkipExec()) {
        acc_.TruncStore<VTA_OUT_WIDTH>(op, dram_);
      }
    } else {
      LOG(FATAL) << "Store do not support memory_type="
                 << op->memory_type;
    }
  }

  // With binary the elements are packed bits (VTA_OPCODE_GEMM_BINARY)
  void RunGEMM(const VTAGemInsn* op, bool binary) {
    if (!op->reset_reg) {
      prof_->gemm_counter += op->iter_out * op->iter_in * (op->uop_end - op->uop_bgn);
      if (prof_->SkipExec()) return;
      for (uint32_t y = 0; y < op->iter_out; ++y) {
        for (uint32_t x = 0; x < op->iter_in; ++x) {
          for (uint32_t uindex = op->uop_bgn; uindex < op->uop_end; ++uindex) {
            VTAUop* uop_ptr = static_cast<VTAUop*>(uop_.BeginPtr(uindex));
            // Read in memory indices
            uint32_t acc_idx = uop_ptr->dst_idx;
            uint32_t inp_idx = uop_ptr->src_idx;
            uint32_t wgt_idx = uop_ptr->wgt_idx;

            acc_idx += y * op->dst_factor_out + x * op->dst_factor_in;
            inp_idx += y * op->src_factor_out + x * op->src_factor_in;
            wgt_idx += y * op->wgt_factor_out + x * op->wgt_factor_in;
            BitPacker<VTA_ACC_WIDTH> acc(acc_.BeginPtr(acc_idx));
            BitPacker<VTA_INP_WIDTH> inp(inp_.BeginPtr(inp_idx));
            BitPacker<VTA_WGT_WIDTH> wgt(wgt_.BeginPtr(wgt_idx));

            // gemm loop
            for (uint32_t i = 0; i < VTA_BATCH; ++i) {
              for (uint32_t j = 0; j < VTA_BLOCK_OUT; ++j) {
                uint32_t acc_offset = i * VTA_BLOCK_OUT + j;
                int32_t sum = acc.GetSigned(acc_offset);
                for (uint32_t k = 0; k < VTA_BLOCK_IN; ++k) {
                  int32_t inp_elem = inp.GetSigned(i * VTA_BLOCK_IN + k);
                  if (op->inp_unsigned) {
                    inp_elem &= (1 << VTA_INP_WIDTH) - 1;
                  }
                  int32_t wgt_elem = wgt.GetSigned(j * VTA_BLOCK_IN + k);
                  if (binary) {
                    // The elements are packed bits (1 is +1, 0 is -1), their
                    // dot product is bits - 2 * popcount(input xor weight)
                    CHECK_EQ(VTA_INP_WIDTH, VTA_WGT_WIDTH);
                    uint32_t diff = static_cast<uint32_t>(inp_elem ^ wgt_elem) &
                        ((1u << VTA_INP_WIDTH) - 1);
                    int32_t different = 0;
                    for (; diff != 0; diff >>= 1) {
                      different += diff & 1;
                    }
                    sum += VTA_INP_WIDTH - 2 * different;
                  } else {
                    sum += inp_elem * wgt_elem;
                  }
                }
                acc.SetSigned(acc_offset, sum);
              }
            }
          }
        }
      }
    } else {
      if (prof_->SkipExec()) return;
      // reset
      for (uint32_t y = 0; y < op->iter_out; ++y) {
        for (uint32_t x = 0; x < op->iter_in; ++x) {
          for (uint32_t uindex = op->uop_bgn; uindex < op->uop_end; ++uindex) {
            VTAUop* uop_ptr = static_cast<VTAUop*>(uop_.BeginPtr(uindex));
            uint32_t acc_idx = uop_ptr->dst_idx;
            acc_idx += y * op->dst_factor_out + x * op->dst_factor_in;
            BitPacker<VTA_ACC_WIDTH> acc(acc_.BeginPtr(acc_idx));
            for (uint32_t i = 0; i < VTA_BATCH * VTA_BLOCK_OUT; ++i) {
              acc.SetSigned(i, 0);
            }
          }
        }
      }
    }
  }

  void RunALU(const VTAAluInsn* op) {
    // The requantization always takes the multiplier from the tensor, the
    // immediate holds its other arguments
    if (op->use_imm && op->alu_opcode != VTA_ALU_OPCODE_REQUANT) {
      RunALU_<true>(op);
    } else {
      RunALU_<false>(op);
    }
  }

  template<bool use_imm>
  void RunALU_(const VTAAluInsn* op) {
    switch (op->alu_opcode) {
      case VTA_ALU_OPCODE_ADD: {
        return RunALULoop<use_imm>(op, [](int32_t x, int32_t y) {
            return x + y;
          });
      }
      case VTA_ALU_OPCODE_MAX: {
        return RunALULoop<use_imm>(op, [](int32_t x, int32_t y) {
            return std::max(x, y);
          });
      }
      case VTA_ALU_OPCODE_MIN: {
        return RunALULoop<use_imm>(op, [](int32_t x, int32_t y) {
            return std::min(x, y);
          });
      }
      case VTA_ALU_OPCODE_SHR: {
        return RunALULoop<use_imm>(op, [](int32_t x, int32_t y) {
            if (y >= 0) {
              return x >> y;
            } else {
              return x << (-y);
            }
          });
      }
      case VTA_ALU_OPCODE_MUL: {
        return RunALULoop<use_imm>(op, [](int32_t x, int32_t y) {
            return x * y;
          });
      }
      case VTA_ALU_OPCODE_REQUANT: {
        // MultiplyByQuantizedMultiplier of TFLite with TFLITE_SINGLE_ROUNDING,
        // optionally truncating or rounding the ties to even (HAWQ-V3)
        uint32_t imm = static_cast<uint32_t>(op->imm);
        int shift = imm & ((1 << VTA_REQUANT_SHIFT_BIT_WIDTH) - 1);
        bool round = (imm >> VTA_REQUANT_ROUND_BIT) & 1;
        bool even = (imm >> VTA_REQUANT_EVEN_BIT) & 1;
        return RunALULoop<false>(op, [shift, round, even](int32_t x, int32_t y) {
            int64_t prod = static_cast<int64_t>(x) * static_cast<int64_t>(y);
            int64_t half = (round && shift != 0) ? static_cast<int64_t>(1) << (shift - 1) : 0;
            int64_t result = (prod + half) >> shift;
            int64_t discarded = prod & ((static_cast<int64_t>(1) << shift) - 1);
            if (even && half != 0 && discarded == half) {
              result &= ~static_cast<int64_t>(1);
            }
            return static_cast<int32_t>(result);
          });
      }
      case VTA_ALU_OPCODE_PACK_INT4: {
        return RunALUPackInt4(op);
      }
      case VTA_ALU_OPCODE_PACK_SIGN: {
        // Shift left by one and shift in 1 if the source is >= 0
        return RunALULoop<use_imm>(op, [](int32_t x, int32_t y) {
            return static_cast<int32_t>(
                (static_cast<uint32_t>(x) << 1) | (y >= 0 ? 1u : 0u));
          });
      }
      default: {
        LOG(FATAL) << "Unknown ALU code " << op->alu_opcode;
      }
    }
  }

  // Packs the low 4 bits of the elements of the tensors src and src + 1, the
  // low nibble of a byte is the first element. The hardware only writes the
  // output memory, this simulator has no output memory (the store truncates
  // the accumulator) so here the packed bytes are written in the tensor dst.
  void RunALUPackInt4(const VTAAluInsn* op) {
    prof_->alu_counter += op->iter_out * op->iter_in * (op->uop_end - op->uop_bgn);
    if (prof_->SkipExec()) return;
    for (int y = 0; y < op->iter_out; ++y) {
      for (int x = 0; x < op->iter_in; ++x) {
        for (int k = op->uop_bgn; k < op->uop_end; ++k) {
          VTAUop* uop_ptr = static_cast<VTAUop*>(uop_.BeginPtr(k));
          uint32_t dst_index = uop_ptr->dst_idx;
          uint32_t src_index = uop_ptr->src_idx;
          dst_index += y * op->dst_factor_out + x * op->dst_factor_in;
          src_index += y * op->src_factor_out + x * op->src_factor_in;
          BitPacker<VTA_ACC_WIDTH> dst(acc_.BeginPtr(dst_index));
          BitPacker<VTA_ACC_WIDTH> src0(acc_.BeginPtr(src_index));
          BitPacker<VTA_ACC_WIDTH> src1(acc_.BeginPtr(src_index + 1));
          int32_t packed[VTA_BATCH * VTA_BLOCK_OUT];
          for (int i = 0; i < VTA_BATCH; ++i) {
            for (int b = 0; b < VTA_BLOCK_OUT; ++b) {
              int lane = i * VTA_BLOCK_OUT + 2 * (b % (VTA_BLOCK_OUT / 2));
              bool second = b >= VTA_BLOCK_OUT / 2;
              int32_t lo = (second ? src1.GetSigned(lane) : src0.GetSigned(lane)) & 0xF;
              int32_t hi = (second ? src1.GetSigned(lane + 1) : src0.GetSigned(lane + 1)) & 0xF;
              packed[i * VTA_BLOCK_OUT + b] = static_cast<int8_t>((hi << 4) | lo);
            }
          }
          for (int n = 0; n < VTA_BATCH * VTA_BLOCK_OUT; ++n) {
            dst.SetSigned(n, packed[n]);
          }
        }
      }
    }
  }

  template<bool use_imm, typename F>
  void RunALULoop(const VTAAluInsn* op, F func) {
    prof_->alu_counter += op->iter_out * op->iter_in * (op->uop_end - op->uop_bgn);
    if (prof_->SkipExec()) return;
    for (int y = 0; y < op->iter_out; ++y) {
      for (int x = 0; x < op->iter_in; ++x) {
        for (int k = op->uop_bgn; k < op->uop_end; ++k) {
          // Read micro op
          VTAUop* uop_ptr = static_cast<VTAUop*>(uop_.BeginPtr(k));
          uint32_t dst_index = uop_ptr->dst_idx;
          uint32_t src_index = uop_ptr->src_idx;
          dst_index += y * op->dst_factor_out + x * op->dst_factor_in;
          src_index += y * op->src_factor_out + x * op->src_factor_in;
          BitPacker<VTA_ACC_WIDTH> dst(acc_.BeginPtr(dst_index));
          BitPacker<VTA_ACC_WIDTH> src(acc_.BeginPtr(src_index));
          for (int k = 0; k < VTA_BATCH * VTA_BLOCK_OUT; ++k) {
            if (use_imm) {
              dst.SetSigned(k, func(dst.GetSigned(k), op->imm));
            } else {
              dst.SetSigned(k, func(dst.GetSigned(k), src.GetSigned(k)));
            }
          }
        }
      }
    }
  }
  // the finish counter
  int finish_counter_{0};
  // Prof_
  Profiler* prof_;
  // The DRAM interface
  DRAM* dram_;
  TlppVerify* ptlpp;
  // The SRAM
  SRAM<VTA_INP_WIDTH, VTA_BATCH * VTA_BLOCK_IN, VTA_INP_BUFF_DEPTH> inp_;
  SRAM<VTA_WGT_WIDTH, VTA_BLOCK_IN * VTA_BLOCK_OUT, VTA_WGT_BUFF_DEPTH> wgt_;
  SRAM<VTA_ACC_WIDTH, VTA_BATCH * VTA_BLOCK_OUT, VTA_ACC_BUFF_DEPTH> acc_;
  SRAM<VTA_UOP_WIDTH, 1, VTA_UOP_BUFF_DEPTH> uop_;
};

using tvm::runtime::TVMRetValue;
using tvm::runtime::TVMArgs;

TVM_REGISTER_GLOBAL("vta.simulator.profiler_clear")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Profiler::ThreadLocal()->Clear();
  });
TVM_REGISTER_GLOBAL("vta.simulator.profiler_status")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = Profiler::ThreadLocal()->AsJSON();
  });
TVM_REGISTER_GLOBAL("vta.simulator.profiler_debug_mode")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Profiler::ThreadLocal()->debug_flag = args[0];
  });
}  // namespace sim
}  // namespace vta

void* VTAMemAlloc(size_t size, int cached) {
  return vta::sim::DRAM::Global()->Alloc(size);
}

void VTAMemFree(void* buf) {
  vta::sim::DRAM::Global()->Free(buf);
}

vta_phy_addr_t VTAMemGetPhyAddr(void* buf) {
  return vta::sim::DRAM::Global()->GetPhyAddr(buf);
}

void VTAMemCopyFromHost(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

void VTAMemCopyToHost(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

void VTAFlushCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
}

void VTAInvalidateCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
}

VTADeviceHandle VTADeviceAlloc() {
  return new vta::sim::Device();
}

void VTADeviceFree(VTADeviceHandle handle) {
  delete static_cast<vta::sim::Device*>(handle);
}

int VTADeviceRun(VTADeviceHandle handle,
                 vta_phy_addr_t insn_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles) {
  return static_cast<vta::sim::Device*>(handle)->Run(
      insn_phy_addr, insn_count, wait_cycles);
}

void VTAProgram(const char* bitstream) {
}
