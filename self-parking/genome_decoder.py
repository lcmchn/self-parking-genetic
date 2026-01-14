import math
from typing import List, Literal, Tuple

# 类型别名
Bit = Literal[0, 1]
Bits = List[Bit]

# 配置常量
CAR_SENSORS_NUM = 8
BIAS_UNITS = 1
MARGIN = 0.4

# 精度配置（固定使用 custom）
PRECISION_CONFIG = {
    "signBitsCount": 1,
    "exponentBitsCount": 4,
    "fractionBitsCount": 5,
    "totalBitsCount": 10,
}

GENES_PER_NUMBER = PRECISION_CONFIG["totalBitsCount"]
COEFFICIENTS_LENGTH = CAR_SENSORS_NUM + BIAS_UNITS

class GenomeDecoder:
    """
    基因组解码器：将二进制基因序列解码为控制信号（转向、油门）

    支持自定义浮点格式（1位符号 + 4位指数 + 5位小数 = 10位）
    """

    def __init__(self):
        self.precision_config = PRECISION_CONFIG
        self.coefficients_length = COEFFICIENTS_LENGTH
        self.genes_per_number = GENES_PER_NUMBER

    @staticmethod
    def _sigmoid(x: float) -> float:
        """安全 Sigmoid 函数，防止溢出"""
        if x < -709:
            return 0.0
        elif x > 709:
            return 1.0
        else:
            return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _sigmoid_to_muscle(sigmoid_val: float) -> int:
        """将 Sigmoid 输出映射为肌肉信号：-1, 0, +1"""
        if sigmoid_val < 0.5 - MARGIN:
            return -1
        elif sigmoid_val > 0.5 + MARGIN:
            return 1
        else:
            return 0

    def _bits_to_float(self, bits: Bits) -> float:
        """将 10 位自定义浮点二进制转为 Python float"""
        cfg = self.precision_config
        sign_bit = bits[0]
        exponent_bits = bits[1:1 + cfg["exponentBitsCount"]]
        fraction_bits = bits[1 + cfg["exponentBitsCount"]:]

        # 符号
        sign = -1 if sign_bit == 1 else 1

        # 指数（带偏置）
        exponent_bias = (1 << (cfg["exponentBitsCount"] - 1)) - 1
        exponent_unbiased = sum(
            bit << (len(exponent_bits) - i - 1)
            for i, bit in enumerate(exponent_bits)
        )
        exponent = exponent_unbiased - exponent_bias

        # 小数部分（隐含前导 1）
        fraction = 1.0 + sum(
            bit * (2 ** -(i + 1))
            for i, bit in enumerate(fraction_bits)
        )

        return sign * (2 ** exponent) * fraction

    def _linear_polynomial(self, coefficients: List[float], variables: List[float]) -> float:
        """计算线性多项式：y = w0*x0 + w1*x1 + ... + b"""
        if len(coefficients) != len(variables) + 1:
            raise ValueError("系数数量应为变量数 + 1（含偏置）")
        return sum(w * x for w, x in zip(coefficients[:-1], variables)) + coefficients[-1]

    def decode(self, genome: Bits, sensors: List[float]) -> Tuple[int, int]:
        """
        解码基因组，生成控制信号

        Args:
            genome: 二进制基因序列，长度必须为 2 * COEFFICIENTS_LENGTH * GENES_PER_NUMBER
            sensors: 8 个传感器输入值（float 列表，长度=8）

        Returns:
            (wheel_signal, engine_signal) -> 每个信号 ∈ {-1, 0, +1}
        """
        if len(sensors) != CAR_SENSORS_NUM:
            raise ValueError(f"传感器数量必须为 {CAR_SENSORS_NUM}")

        expected_genome_length = 2 * self.coefficients_length * self.genes_per_number
        if len(genome) != expected_genome_length:
            raise ValueError(f"基因组长度应为 {expected_genome_length}，实际为 {len(genome)}")

        # 解码轮子（转向）系数
        wheel_coeffs = []
        for i in range(self.coefficients_length):
            start = i * self.genes_per_number
            bits = genome[start:start + self.genes_per_number]
            wheel_coeffs.append(self._bits_to_float(bits))

        # 解码引擎（油门）系数
        engine_coeffs = []
        offset = self.coefficients_length * self.genes_per_number
        for i in range(self.coefficients_length):
            start = offset + i * self.genes_per_number
            bits = genome[start:start + self.genes_per_number]
            engine_coeffs.append(self._bits_to_float(bits))

        # 计算神经输出
        wheel_output = self._linear_polynomial(wheel_coeffs, sensors)
        engine_output = self._linear_polynomial(engine_coeffs, sensors)

        # 转为肌肉信号
        wheel_signal = self._sigmoid_to_muscle(self._sigmoid(wheel_output))
        engine_signal = self._sigmoid_to_muscle(self._sigmoid(engine_output))

        return wheel_signal, engine_signal


# ==================== 使用示例 ====================
if __name__ == "__main__":
    decoder = GenomeDecoder()

    # 模拟 8 个传感器值（单位：米）
    sensor_values = [1.2, 0.8, 3.5, 0.0, 2.1, 4.0, 0.3, 1.9]

    # 构造一个合法的基因组（全零，仅用于测试长度）
    genome_length = 2 * COEFFICIENTS_LENGTH * GENES_PER_NUMBER  # = 2 * 9 * 10 = 180
    dummy_genome = [0] * genome_length

    wheel, engine = decoder.decode(dummy_genome, sensor_values)
    print(f"控制信号 -> 转向: {wheel}, 油门: {engine}")