from dataclasses import asdict, dataclass

import config as cfg


@dataclass(slots=True)
class ExperimentRequest:

    # ------------------------------------------------------------------
    # Core experiment config
    # ------------------------------------------------------------------
    preset: cfg.Preset = cfg.DEFAULT_PRESET

    arch: str = cfg.ARCH

    num_rounds: int = cfg.NUM_ROUNDS
    num_clients: int = cfg.NUM_CLIENTS
    num_malicious: int = cfg.NUM_MALICIOUS

    iid_rate: float = cfg.IID_RATE

    defense: str = "safesplit"
    backdoor: str = cfg.BACKDOOR_TYPE

    # ------------------------------------------------------------------
    # Attack config
    # ------------------------------------------------------------------
    pdr: float = cfg.POISONED_DATA_RATE

    attack_schedule: str = cfg.ATTACK_SCHEDULE

    slow_pdr_start: float = cfg.SLOW_POISON_START_PDR
    slow_pdr_end: float = cfg.SLOW_POISON_END_PDR
    slow_ramp_rounds: int = cfg.SLOW_POISON_RAMP_ROUNDS

    # ------------------------------------------------------------------
    # Runtime config
    # ------------------------------------------------------------------
    device: str | None = None

    seed: int = cfg.SEED

    out_dir: str = str(cfg.RESULTS_DIR)

    write_json: bool = True

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    local_epochs: int = cfg.LOCAL_EPOCHS

    batch_size: int = cfg.BATCH_SIZE
    eval_batch_size: int = cfg.EVAL_BATCH_SIZE

    max_samples_per_client: int | None = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        preset: cfg.Preset | str | None = None,
        **overrides,
    ):

        resolved_preset = cfg.normalize_preset(preset)

        preset_values = cfg.resolve_preset(resolved_preset)

        self.preset = resolved_preset

        self.arch = preset_values.arch

        self.num_rounds = preset_values.num_rounds
        self.num_clients = preset_values.num_clients
        self.num_malicious = preset_values.num_malicious

        self.iid_rate = preset_values.iid_rate

        self.defense = "safesplit"
        self.backdoor = cfg.BACKDOOR_TYPE

        self.pdr = cfg.POISONED_DATA_RATE

        self.attack_schedule = cfg.ATTACK_SCHEDULE

        self.slow_pdr_start = cfg.SLOW_POISON_START_PDR
        self.slow_pdr_end = cfg.SLOW_POISON_END_PDR
        self.slow_ramp_rounds = preset_values.num_rounds

        self.device = None

        self.seed = cfg.SEED

        self.out_dir = str(cfg.RESULTS_DIR)

        self.write_json = True

        self.local_epochs = preset_values.local_epochs

        self.batch_size = preset_values.batch_size
        self.eval_batch_size = preset_values.eval_batch_size

        self.max_samples_per_client = (
            preset_values.max_samples_per_client
        )

        # --------------------------------------------------------------
        # Apply user overrides
        # --------------------------------------------------------------
        for key, value in overrides.items():

            if value is None:
                raise ValueError(f"ExperimentRequest does not allow None values (got None for '{key}')")

            if not hasattr(self, key):
                raise ValueError(f"Unknown ExperimentRequest field: '{key}'")

            setattr(self, key, value)

    # ------------------------------------------------------------------
    # Serialization helper
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, object]:

        data = asdict(self)

        data["preset"] = self.preset.value

        return data
