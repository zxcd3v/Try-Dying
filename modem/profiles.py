"""Все настраиваемые параметры сигнала — только здесь (тюнингу не расползаться).

Термины — CONTEXT.md; решения — ADR-0001 (MFSK, тона строго в FFT-бины),
ADR-0002 (самодостаточный аир-кадр, аир-заголовок ×3 + CRC-8),
ADR-0003 (полосы: позиции сетки тонов в рабочем окне).

Ширина FFT-бина окна символа = SAMPLE_RATE / symbol_samples; каждый тон
обязан быть кратен ей, иначе энергия размазывается по соседним бинам.
"""

from dataclasses import dataclass, replace

SAMPLE_RATE = 48_000  # Гц, mono, float32 — контракт v1.2

# Длинный chirp теряет корреляцию уже при ~0.05% масштаба времени, поэтому
# приёмник ищет преамбулу банком шаблонов по сетке дрейфов (тикет 03).
# Крайние точки — критерий рассинхрона часов ±0.1%; шаг 0.05% держит потерю
# пика на худшем промежуточном дрейфе в пределах ~0.8 от идеального.
_DRIFT_GRID = (-0.001, -0.0005, 0.0, 0.0005, 0.001)


@dataclass(frozen=True)
class Profile:
    """Именованный набор параметров сигнала одного профиля."""

    max_block: int               # предел длины блока, байт (контракт v1.2)
    tones_hz: tuple[float, ...]  # сетка тонов; номер тона = значение символа
    symbol_samples: int          # окно символа (оно же окно FFT на приёме)
    guard_samples: int           # защитный интервал после символа (эхо)
    ramp_samples: int            # плавный фронт/спад символа (против щелчков)
    preamble_samples: int        # длительность chirp-преамбулы
    chirp_lo_hz: float           # начало частотного свипа преамбулы
    chirp_hi_hz: float           # конец свипа
    preamble_gap_samples: int    # тишина между преамбулой и аир-заголовком
    frame_gap_samples: int       # тишина между соседними аир-кадрами
    pilot_every: int             # пилот-паттерн после каждых N символов данных
    pilot_pattern: tuple[int, ...]  # значения символов пилота (известны заранее)
    header_repeats: int          # аир-заголовок повторяется ×N, приём по большинству
    amplitude: float             # пиковая амплитуда сигнала, 0..1
    preamble_threshold: float    # порог норм. корреляции [-1..1] находки преамбулы
    drift_grid: tuple[float, ...]  # сетка дрейфов банка шаблонов преамбулы rx
    subcarriers: int = 1         # параллельных MFSK-поднесущих в символе (тикет 09)

    @property
    def tones_per_carrier(self) -> int:
        """Сетка tones_hz делится поровну: свой кусок каждой поднесущей."""
        return len(self.tones_hz) // self.subcarriers

    @property
    def bits_per_carrier(self) -> int:
        return (self.tones_per_carrier - 1).bit_length()

    @property
    def bits_per_symbol(self) -> int:
        return self.subcarriers * self.bits_per_carrier

    @property
    def slot_samples(self) -> int:
        """Шаг сетки символов: окно символа + защитный интервал."""
        return self.symbol_samples + self.guard_samples

    def symbols_for_bytes(self, n_bytes: int) -> int:
        """Сколько символов занимают n байтов (хвост добит нулями)."""
        return -(-8 * n_bytes // self.bits_per_symbol)

    def chord_tones(self, value: int) -> tuple[float, ...]:
        """Частоты аккорда символа: по одному тону на поднесущую.

        Старшие биты значения — нижняя по частоте поднесущая (зеркало —
        rx._symbol_at). При subcarriers=1 аккорд вырождается в один тон.
        """
        per, bits = self.tones_per_carrier, self.bits_per_carrier
        mask = per - 1
        return tuple(
            self.tones_hz[c * per + ((value >> (bits * (self.subcarriers - 1 - c))) & mask)]
            for c in range(self.subcarriers)
        )


PROFILES = {
    # Максимальный шум, якорная полоса ~4–7 кГц: 4 тона далеко друг от друга
    # (800 Гц = 8 бинов), длинный символ 10 мс, щедрый защитный интервал.
    "robust": Profile(
        max_block=96,
        tones_hz=(4_500.0, 5_300.0, 6_100.0, 6_900.0),  # бин 100 Гц: 45/53/61/69
        symbol_samples=480,       # 10 мс → бин 100 Гц
        guard_samples=120,        # 2.5 мс на затухание эха
        ramp_samples=48,          # 1 мс
        preamble_samples=4_800,   # 100 мс chirp — уверенная корреляция в шуме
        chirp_lo_hz=2_000.0,
        chirp_hi_hz=8_000.0,
        preamble_gap_samples=240,
        frame_gap_samples=2_400,  # 50 мс между аир-кадрами
        pilot_every=32,
        pilot_pattern=(0, 3),     # крайние тона — легко отличить от шума
        header_repeats=3,
        amplitude=0.8,
        # Тюнинг тикета 03: пик НКК на чистом канале ~1.0, при SNR −9 дБ
        # падает до ~0.3; шумовой фон НКК (max по ~10^5 офсетов) ~0.07 —
        # 0.25 берёт слабые преамбулы, не ловя шум.
        preamble_threshold=0.25,
        drift_grid=_DRIFT_GRID,
    ),
    # Обычная аудитория: 16-MFSK (символ = hex-цифра). Тюнинг тикета 08:
    # символ 160 сэмплов (бин 300 Гц) и шаг сетки в 1 бин — иначе 16 тонов
    # с двухбинным шагом не влезают в аппаратное окно 1.5–9.5 кГц, а более
    # длинный символ не даёт критерий скорости ≥5× robust. Сетка 2.4–6.9 кГц
    # сидит в середине окна АЧХ, вдали от краёв тракта.
    "medium": Profile(
        max_block=128,
        tones_hz=tuple(2_400.0 + 300.0 * k for k in range(16)),  # 2.4–6.9 кГц
        symbol_samples=160,       # 3.3 мс → бин 300 Гц
        guard_samples=40,
        ramp_samples=16,
        preamble_samples=4_800,
        chirp_lo_hz=2_000.0,
        chirp_hi_hz=8_000.0,
        preamble_gap_samples=240,
        frame_gap_samples=1_200,
        pilot_every=48,
        pilot_pattern=(0, 15),
        header_repeats=3,
        amplitude=0.8,
        # Тюнинг тикета 08 (разведка: `python tools/snr_grid.py medium`):
        # порог тот же, что у robust, — при поле medium −3 дБ преамбула
        # находится с большим запасом, ложных срабатываний на шуме нет.
        preamble_threshold=0.25,
        drift_grid=_DRIFT_GRID,
    ),
    # Тишина, максимум скорости: 9 параллельных 4-MFSK-поднесущих
    # («OFDM-лайт», тикет 09) — символ-аккорд несёт 18 бит. Сетка 2.0–9.0 кГц
    # шагом 200 Гц (бин окна 240 сэмплов): по 4 соседних бина на поднесущую.
    # 4 тона на поднесущую, а не 16: b бит на 2^b бинов максимален при b≤2,
    # а 4-MFSK различается в шуме лучше двоичного при том же счёте бит.
    "fast": Profile(
        max_block=255,  # предел кодового слова Reed-Solomon (GF(256))
        tones_hz=tuple(2_000.0 + 200.0 * k for k in range(36)),  # 2.0–9.0 кГц
        symbol_samples=240,       # 5 мс → бин 200 Гц
        guard_samples=40,
        ramp_samples=16,
        preamble_samples=2_400,
        chirp_lo_hz=1_500.0,
        chirp_hi_hz=8_500.0,
        # Эхо полноамплитудной преамбулы давит аккорды, которые в 9 раз
        # тише каждого тона: заголовок прячется за паузу длиннее критерия
        # эха (50 мс), иначе кадр гибнет целиком (разведка тикета 09).
        preamble_gap_samples=2_640,
        frame_gap_samples=960,
        pilot_every=64,
        pilot_pattern=(0, 2**18 - 1),  # все поднесущие тоном 0 / тоном 3
        header_repeats=3,
        amplitude=0.8,
        preamble_threshold=0.5,
        drift_grid=_DRIFT_GRID,
        subcarriers=9,
    ),
}

# Экспорт контракта v1.2; истина о пределе — в самом профиле.
MAX_BLOCK = {name: profile.max_block for name, profile in PROFILES.items()}


# --- полосы (ADR-0003) ---------------------------------------------------
#
# Полоса — позиция сетки тонов профиля внутри рабочего окна 1.5–9.5 кГц.
# ADR-0003 писался до тюнинга тикетов 08/09 и обещал «16 тонов шагом 125 Гц»;
# после тюнинга шаг medium — 300 Гц, а сетка fast занимает окно почти целиком,
# поэтому полосы строятся сдвигом готовой сетки, а не её пересборкой. Суть
# решения та же: 3–4 полосы, один зонд, одно измерение сразу на все.

WINDOW_LO_HZ = 1_500.0  # аппаратное окно тракта «динамик → комната → микрофон»
WINDOW_HI_HZ = 9_500.0

BAND_COUNT = 4  # «3–4 полосы» ADR-0003; дедупликация может оставить меньше
ANCHOR = "robust"  # якорная полоса — служебный канал, калибровке не подлежит


@dataclass(frozen=True)
class Band:
    """Одна позиция сетки тонов профиля в рабочем окне."""

    index: int
    offset_hz: float
    lo_hz: float
    hi_hz: float

    @property
    def label(self) -> str:
        return f"{self.lo_hz:.0f}–{self.hi_hz:.0f} Гц"


def bands(profile_name: str) -> tuple[Band, ...]:
    """Полосы профиля по возрастанию частоты. У якоря она всегда одна."""
    if profile_name not in _BANDS:
        raise ValueError(f"неизвестный профиль {profile_name!r}")
    return _BANDS[profile_name]


def default_band(profile_name: str) -> int:
    """Полоса по умолчанию — сетка профиля как есть (смещение 0).

    На ней сняты пороги SNR и таблица скоростей, поэтому она же — фолбэк
    при несостоявшейся калибровке (ADR-0003).
    """
    return next(b.index for b in bands(profile_name) if b.offset_hz == 0.0)


def banded(profile_name: str, band: int | None = None) -> Profile:
    """Профиль со сдвинутой в выбранную полосу сеткой тонов."""
    if band is None:
        band = default_band(profile_name)
    elif not 0 <= band < len(bands(profile_name)):
        raise ValueError(
            f"полоса {band} вне диапазона профиля {profile_name!r}: "
            f"0..{len(bands(profile_name)) - 1}"
        )
    chosen = bands(profile_name)[band]
    profile = PROFILES[profile_name]
    if chosen.offset_hz == 0.0:
        return profile
    return replace(
        profile, tones_hz=tuple(t + chosen.offset_hz for t in profile.tones_hz)
    )


def _band_offsets_hz(profile_name: str) -> tuple[float, ...]:
    """Смещения сетки: нуль плюс равномерный веер по остатку окна.

    Запас считается в бинах и округляется вниз — сетка обязана целиком
    остаться внутри окна, а каждый тон — попасть точно в бин (ADR-0001).
    Ближайшая к нулю точка веера заменяется самим нулём: сетка профиля как
    есть должна быть среди полос всегда.
    """
    profile = PROFILES[profile_name]
    if profile_name == ANCHOR:
        return (0.0,)
    bin_hz = SAMPLE_RATE / profile.symbol_samples
    down = int((min(profile.tones_hz) - WINDOW_LO_HZ) // bin_hz)
    up = int((WINDOW_HI_HZ - max(profile.tones_hz)) // bin_hz)
    steps = [round(-down + (up + down) * i / (BAND_COUNT - 1)) for i in range(BAND_COUNT)]
    steps[min(range(len(steps)), key=lambda i: abs(steps[i]))] = 0
    return tuple(step * bin_hz for step in sorted(set(steps)))


_BANDS = {
    name: tuple(
        Band(
            index=i,
            offset_hz=off,
            lo_hz=min(PROFILES[name].tones_hz) + off,
            hi_hz=max(PROFILES[name].tones_hz) + off,
        )
        for i, off in enumerate(_band_offsets_hz(name))
    )
    for name in PROFILES
}
