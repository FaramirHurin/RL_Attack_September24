# -----------------------------------------------------------------------------#
# Cardsim: A Bayesian simulator for payment card fraud detection research
# Author: Jeff Allen
# -----------------------------------------------------------------------------#
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import lognorm, triang
from sklearn.preprocessing import MinMaxScaler

from banksys import Payer, Terminal, Transaction


def measure_duration():
    """Decorator that measures and prints the execution time of a function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"{func.__name__} executed in {duration:.2f} seconds")
            return result

        return wrapper

    return decorator


class Cardsim:
    """
    A class for simulating payment card transactions and fraud.

    Attributes
    ----------
    DEFAULT_DISTANCE_MODE_QUANTILE : dict
        The quantiles to find the modes in the triangular distribution
        for the distance of in-person, remote, and fraudulent payments.
        The triangular distribution is applied to the payee indices
        after sorting them in ascending order by distance for each payer.
    DEFAULT_MARGINAL_TOD_WEIGHTS : dict
        Weights for each component of the marginal time of day distribution.
    DEFAULT_MARGINAL_TOD_WINDOWS : dict
        Start and end hours for peak windows in the marginal time of day
        distribution.
    DEFAULT_CONDITIONAL_TOD_WEIGHTS : dict
        Weights for each component of the conditional time of day distribution.
    DEFAULT_CONDITIONAL_TOD_WINDOWS : dict
        Start and end hours for peak windows in the conditional time of day
        distribution.
    """

    DEFAULT_DISTANCE_MODE_QUANTILE = {"in_person": 0.01, "remote": 0.5, "fraud": 0.75}

    DEFAULT_MARGINAL_TOD_WEIGHTS = {"baseline": 0.4, "breakfast": 0.2, "lunch": 0.2, "dinner": 0.2}

    DEFAULT_MARGINAL_TOD_WINDOWS = {"breakfast": (7, 9), "lunch": (11, 13), "dinner": (18, 21)}

    DEFAULT_CONDITIONAL_TOD_WEIGHTS = {
        "baseline": 0.5,
        "night": 0.3,
        "morning": 0.2,
    }

    DEFAULT_CONDITIONAL_TOD_WINDOWS = {"night": (22, 24), "morning": (0, 5)}

    def __init__(
        self,
        seed: Optional[int] = None,
        dcpc_start_year: int = 2022,
        dcpc_end_year: int = 2023,
        dcpc_folder: Optional[str] = None,
        txns_samples_m: int = 2500,
        txns_samples_n: int = 100,
        value_samples_m: int = 5000,
        value_samples_n: int = 200,
        grid_size: int = 200,
        payer_payee_factor: int = 10,
        debit_fraud_mult: float = 2.0,
        credit_fraud_mult: float = 1.45,
        credit_card_marginal_p: float = 0.38,
        credit_card_conditional_p: float = 0.57,
        remote_marginal_p: float = 0.36,
        remote_conditional_p: float = 0.63,
        distance_mode_quantile=None,
        marginal_tod_weights=None,
        marginal_tod_windows=None,
        conditional_tod_weights=None,
        conditional_tod_windows=None,
        tod_smoothing_param: Optional[float] = 0.5,
        fraud_rate: float = 0.01,
        lr_cap: float = 5.0,
        fraud_flag_threshold: float = 0.01,
        addnoise: bool = False,
    ):
        """
        Create a payment transaction simulator.

        Parameters
        ----------
        log_level: str, optional
            Threshold for the logger. The default is 'INFO'.
        seed : int, optional
            A seed for reproducibility. The default is None.
        dcpc_start_year : int, optional
            The first year of data to use from the Diary of Consumer Payment
            Choice (DCPC) in simulating payer characteristics. The default is
            2022.
        dcpc_end_year : int, optional
            The last year of data to use from the Diary of Consumer Payment
            Choice (DCPC) in simulating payer characteristics. The default is
            2023.
        dcpc_folder : str or None
            A folder where the DCPC data live. If None, data are sourced from
            FRB Atlanta website. The default is None. The system needs at least
            one year of DCPC data and files for the following DCPC levels:
            day, ind, tran. An example naming structure is:
            dcpc_2023_daylevel_public_xls.csv. Cardsim ships with two years
            of data in cardsim/dcpc. The vignette shows how to access the
            folder path. The simulator is faster when loading files locally.
        txns_samples_m : int, optional
            The number of samples to draw when simulating average number of
            daily payment transactions. Increasing the number smooths out the
            curve. The default is 2500.
        txns_samples_n : int, optional
            The size of the samples when simulating average number of
            daily payment transactions. Increasing this number decreases the
            variance. Some variance is good. The default is 100.
        value_samples_m : int, optional
            The number of samples to draw when simulating average values.
            Increasing the number smooths out the curve. The default is 5000.
        value_samples_n : int, optional
            The size of the samples when simulating average values. Increasing
            this number decreases the variance. Some variance is good. The
            default is 200.
        grid_size : int, optional
            The size (x, y) of the grid in which payers and payees reside.
            For example, grid_size=100 will create a 100 X 100 grid. The
            default is 200.
        payer_payee_factor : int, optional
            Payer to payees factor. Used as a divisor in calculating number of
            payees based on the number of payers. The default is 10. Derived
            from DCPC and Census Bureau data.
        debit_fraud_mult: float, optional
            Multiplier for the average value of fraudulent debit card
            transactions. The default is 2.0. Derived from FRPS.
        credit_fraud_mult: float, optional
            Multiplier for the average value of fraudulent credit card
            transactions. The default is 1.45. Derived from FRPS.
        credit_card_marginal_p: float, optional
            Probability that a card transaction is made with a credit card.
            Debit card probability calculated as 1-credit_card_marginal_p.
            The default is 0.38. Derived from FRPS.
        credit_card_conditional_p: float, optional
            The conditional probability of p(credit card | fraud). Also used
            in calculating debit card conditional conditional probability.
            The default is 0.57. Derived from FRPS.
        remote_marginal_p: float, optional
            Probability that a card transaction is made remotely. In person
            probability calculated as 1-remote_marginal_p. The default is 0.36.
            Derived from FRPS.
        remote_conditional_p: float, optional
            The conditional probability of p(remote | fraud). Also used
            in calculating in person conditional probability. The default is
            0.63. Derived from FRPS.
        distance_mode_quantile: dict or None, optional
            The quantiles to find the modes in the triangular distribution
            for the distance of in-person, remote, and fraudulent payments.
            The triangular distribution is applied to the merchant indices
            after sorting them in ascending order by distance for each payer.
            Defaults to None and inherits from DEFAULT_DISTANCE_MODE_QUANTILE
        marginal_tod_weights: dict or None, optional
            Weights for each component of the marginal time of day distribution.
            Defaults to None and inherits from DEFAULT_MARGINAL_TOD_WEIGHTS.
        marginal_tod_windows : dict or None, optional
            Start and end hours for peak windows in the marginal time of day
            distribution. Defaults to None and inherits from
            DEFAULT_MARGINAL_TOD_WINDOWS.
        conditional_tod_weights : dict or None, optional
            Weights for each component of the conditional time of day
            distribution. Defaults to None and inherits from
            DEFAULT_CONDITIONAL_TOD_WEIGHTS.
        conditional_tod_windows : dict or None, optional
            Start and end hours for peak windows in the conditional time of day
            distribution. Defaults to None and inherits from
            DEFAULT_CONDITIONAL_TOD_WINDOWS.
        tod_smoothing_param : float or None, optional
            A parameter between 0-1 to smooth out the hourly probability
            mass function. The default is 0.5. With a parameter of 0.5, the
            smoothed hourly fraud probabilities would be
            (marginal_pmf * 0.5) + (conditional_pmf * (1-0.5)). There is a big
            difference in the likelihood ratios for normal and fraudulent
            transactions during certain hours, causing the Bayesian prediction
            to generate very few or no fraudulent transactions during those
            times, which may not be realistic. The smoothing parameter mitigates
            this behavior.
        fraud_rate: float, optional
            The prior probability of fraud assumed by the simulator.
            The default is 0.01.
        lr_cap: float or int, optional
            A cap for the likelihood ratios. In some rare cases, the likelihood
            ratios are very large. This could occur, for example, if a very high
            value is drawn that is much more likely to be from the fraud
            distribution. A cap ensures that a high ratio doesn't overpower the
            calculation. The default is 5.
        fraud_flag_threshold: float, optional
            The percent threshold to use for the fraud flag odds. Threshold
            labels the top n% of transactions in terms of fraud odds. The
            default is 0.01.
        """
        # Configures logging level for the class
        # Seeds: vary for key simulation components, so data are not identical
        self.addnoise = addnoise

        self.base_seed = seed
        # World
        self.dcpc_start_year = dcpc_start_year
        self.dcpc_end_year = dcpc_end_year
        self.dcpc_folder = dcpc_folder
        self.grid_size = grid_size

        self.payer_payee_factor = int(self._jitter(payer_payee_factor, rel_std=0.05, min_val=1))

        self.txns_samples_m = int(self._jitter(txns_samples_m, rel_std=0.02, min_val=1))
        self.txns_samples_n = int(self._jitter(txns_samples_n, rel_std=0.02, min_val=1))
        self.value_samples_m = int(self._jitter(value_samples_m, rel_std=0.02, min_val=1))
        self.value_samples_n = int(self._jitter(value_samples_n, rel_std=0.02, min_val=1))

        self.debit_fraud_mult = self._jitter(debit_fraud_mult, rel_std=0.03, min_val=1.0)
        self.credit_fraud_mult = self._jitter(credit_fraud_mult, rel_std=0.03, min_val=1.0)

        # Simulator pr pyobabilities (clipped)
        self.credit_card_marginal_p = self._jitter(credit_card_marginal_p, rel_std=0.02, min_val=0.0, max_val=1.0)
        self.credit_card_conditional_p = self._jitter(credit_card_conditional_p, rel_std=0.02, min_val=0.0, max_val=1.0)
        self.remote_marginal_p = self._jitter(remote_marginal_p, rel_std=0.02, min_val=0.0, max_val=1.0)
        self.remote_conditional_p = self._jitter(remote_conditional_p, rel_std=0.02, min_val=0.0, max_val=1.0)

        self.distance_mode_quantile = (
            self.DEFAULT_DISTANCE_MODE_QUANTILE.copy() if distance_mode_quantile is None else distance_mode_quantile
        )
        self.marginal_tod_weights = self.DEFAULT_MARGINAL_TOD_WEIGHTS.copy() if marginal_tod_weights is None else marginal_tod_weights
        self.marginal_tod_windows = self.DEFAULT_MARGINAL_TOD_WINDOWS.copy() if marginal_tod_windows is None else marginal_tod_windows
        self.conditional_tod_weights = (
            self.DEFAULT_CONDITIONAL_TOD_WEIGHTS.copy() if conditional_tod_weights is None else conditional_tod_weights
        )
        self.conditional_tod_windows = (
            self.DEFAULT_CONDITIONAL_TOD_WINDOWS.copy() if conditional_tod_windows is None else conditional_tod_windows
        )
        self.tod_smoothing_param = tod_smoothing_param
        # Fraud generation
        self.fraud_rate = fraud_rate
        self.lr_cap = lr_cap
        self.fraud_flag_threshold = fraud_flag_threshold
        self.run_id = None

    def _jitter(self, x, rel_std=0.01, min_val=None, max_val=None):
        """
        Apply small relative Gaussian noise to a scalar if addnoise=True.
        """
        if not self.addnoise or x is None:
            return x

        noisy = x * (1.0 + np.random.normal(0.0, rel_std))

        if min_val is not None or max_val is not None:
            noisy = np.clip(noisy, min_val, max_val)

        return noisy

    @property
    def t_start(self) -> datetime:
        return datetime(self.dcpc_end_year, 1, 1)

    @staticmethod
    def import_dcpc_data(collection: str = "day", start_year: int = 2022, end_year: int = 2023, folder: Optional[str] = None):
        """
        Import diary of consumer payment choice data.

        Parameters
        ----------
        collection : str, optional
            The collection to import. Valid options are 'day', 'tran', and
            'ind'. The default is 'day'.
        start_year : int, optional
            The first year to import. The default is 2022.
        end_year : int, optional
            The last year to import. The default is 2023.
        folder : str or None
            A folder where the DCPC data live. If None, data are sourced from
            FRB Atlanta website. The default is None.

        Returns
        -------
        Pandas dataframe
            A dataframe of DCPC data.

        """
        valid_collections = ["day", "tran", "ind"]

        if collection not in valid_collections:
            raise ValueError(f"Invalid collection. Choose one of: {', '.join(valid_collections)}")

        if collection == "day":
            schema: dict = {
                "id": pl.Int64,
                "date": pl.String,
                "diary_day": pl.Int64,
                "ind_weight": pl.Float64,
                "dow_weight": pl.Float64,
            }
            sort = ["id", "diary_day"]
        elif collection == "tran":
            schema = {
                "id": pl.Int64,
                "diary_day": pl.Int64,
                "tran": pl.Int64,
                "pi": pl.Float64,
                "amnt": pl.Float64,
            }
            sort = ["id", "diary_day", "tran", "pi"]
        else:
            schema = {
                "id": pl.Int64,
                "cc_adopt": pl.Int64,
                "dc_adopt": pl.Int64,
            }
            sort = ["id"]

        dfs = list[pl.DataFrame]()
        for year in range(start_year, end_year + 1):
            # Convert to Path object for OS-agnostic sourcing

            if folder is not None:
                if hasattr(folder, "_paths"):
                    # For MultiplexedPath (conda) and similar
                    folder_path = Path(folder._paths[0])  # type: ignore
                else:
                    folder_path = Path(folder)
                source = folder_path / f"dcpc_{year}_{collection}level_public_xls.csv"
            else:
                # Download the file
                url = f"https://www.atlantafed.org/-/media/documents/banking/consumer-payments/survey-diary-consumer-payment-choice/{year}/dcpc_{year}_{collection}level_public_xls.csv"
                logging.info(f"Retrieving DCPC data from FRB Atlanta website: {url}")
                content = urllib.request.urlopen(url).read()
                source = StringIO(content.decode("utf-8"))
            df = (
                pl.read_csv(source, columns=list(schema.keys()), null_values=["NA"])
                .with_columns(year=pl.lit(year))
                .cast(schema)
                .sort(by=sort)
            )
            dfs.append(df)
        return pl.concat(dfs)

    @measure_duration()
    def source_format_dcpc_data(self):
        """
        Source and format the Diary of Consumer Payment Choice (DCPC) data.
        """
        # Import data ------------------
        logging.info("Sourcing DCPC data")
        cache_file = os.path.join("cache", f"dcpc_ind-{self.dcpc_start_year}-{self.dcpc_end_year}.csv")
        if os.path.exists(cache_file):
            indivs = pl.read_csv(cache_file)
            logging.info("Sourcing of individual data from cache successful")
        else:
            indivs = Cardsim.import_dcpc_data(
                collection="ind",
                start_year=self.dcpc_start_year,
                end_year=self.dcpc_end_year,
                folder=self.dcpc_folder,
            )
            os.makedirs("cache", exist_ok=True)
            indivs.write_csv(cache_file)

        cache_file = os.path.join("cache", f"dcpc_day-{self.dcpc_start_year}-{self.dcpc_end_year}.csv")
        if os.path.exists(cache_file):
            daily = pl.read_csv(cache_file)
            logging.info("Sourcing of daily data from cache successful")
        else:
            daily = Cardsim.import_dcpc_data(
                collection="day",
                start_year=self.dcpc_start_year,
                end_year=self.dcpc_end_year,
                folder=self.dcpc_folder,
            )
            os.makedirs("cache", exist_ok=True)
            daily.write_csv(cache_file)

        cache_file = os.path.join("cache", f"dcpc_trx-{self.dcpc_start_year}-{self.dcpc_end_year}.csv")
        if os.path.exists(cache_file):
            transactions = pl.read_csv(cache_file)
            logging.info("Sourcing of transactions from cache successful")
        else:
            transactions = Cardsim.import_dcpc_data(
                collection="tran",
                start_year=self.dcpc_start_year,
                end_year=self.dcpc_end_year,
                folder=self.dcpc_folder,
            )
            os.makedirs("cache", exist_ok=True)
            transactions.write_csv(cache_file)
        logging.info("Sourcing successful; formatting data")

        indivs = indivs.to_pandas()
        daily = daily.to_pandas()
        transactions = transactions.to_pandas()
        # Format individual data ------------------
        """
        Create indicator for whether respondent has adopted debit or credit 
        card. The code below assigns 0 where both are NA. That is OK here 
        because we will ultimately only retain those who have indicated that 
        they adopted a credit or debit card. 
        """
        indivs["cc_dc"] = np.where((indivs["cc_adopt"] == 1) | (indivs["dc_adopt"] == 1), 1, 0)

        # Format daily data ----------------------
        """
        Remove those that don't have an ind_weight. These are mostly connected
        to respondents from the extra California pool. 
        """
        daily = daily[~daily["ind_weight"].isnull()]

        # Count the number of diary days for id-year combinations
        diary_days = daily.groupby(["id", "year"])["diary_day"].count().reset_index()

        # Identify those who did not participate all 4 days [days 0-3]
        missing_all_days = diary_days["id"][diary_days["diary_day"] < 4].values

        # Drop those who did not participate all 4 days
        daily = daily[~daily["id"].isin(missing_all_days)].reset_index(drop=True)

        """
        Drop those without dow_weight. This mostly corresponds to day 0 and
        those who were assigned to days in September and November, which was
        meant to smooth out issues from diary fatigue. 
        """
        daily = daily[~daily["dow_weight"].isnull()]

        # Merge cardholders
        daily = daily.merge(indivs[["id", "year", "cc_dc"]], how="left", on=["id", "year"])

        # Drop those without a card
        daily = daily[daily["cc_dc"] == 1]

        # Format transactions data ----------------------
        # Need to get the dow_weight on the transactions data
        transactions = transactions.merge(daily, on=["id", "diary_day", "year"], how="left")

        transactions = transactions[~transactions["dow_weight"].isnull()]

        transactions["amnt_w"] = transactions["amnt"] * transactions["dow_weight"]

        card_txns = transactions.loc[transactions["pi"].isin([3.0, 4.0])].copy()

        card_txns["card_type"] = np.where(card_txns["pi"] == 3.0, "Credit", "Debit")

        # Add number of card transactions to daily data ----------------
        """
        First, count the number of card transactions. The recommended unit of
        analysis for transaction data is id-diary_day. We are adding year 
        because we have multiple years. 
        """
        card_txns_count = card_txns.groupby(["id", "diary_day", "year"])["tran"].count().reset_index().rename(columns={"tran": "txns"})

        card_txns_daily = daily.merge(card_txns_count, how="left", on=["id", "diary_day", "year"])

        card_txns_daily["txns"] = np.where(card_txns_daily["txns"].isnull(), 0, card_txns_daily["txns"])

        card_txns_daily["txns_w"] = card_txns_daily["txns"] * card_txns_daily["dow_weight"]

        logging.info("DCPC data sourcing and formatting complete")
        return card_txns, card_txns_daily

    def sample_payments(self, pmnt_series, m, n):
        """
        Draw m samples of size n of a payments series. Used in simulating
        representative payment values and daily number of transactions.

        Parameters
        ----------
        pmnt_series : Pandas series
            A series of transaction values or counts.
        m : int
            The number of samples to generate.
        n : int
            The size of the samples.

        Returns
        -------
        Numpy array
            A numpy array of samples of size m, n
        """
        return np.random.choice(pmnt_series, size=(m, n), replace=True)

    @staticmethod
    def calculate_mad(samples, scaled=True):
        """
        Calculate the median absolute deviation of the simulated transaction
        samples.

        Parameters
        ----------
        samples : Numpy array
            A 2-D numpy array of size (m, n), where m is the number of samples
            and n is the size of each sample.
        scaled : bool
            Whether to use a scaled mad. The default is True.

        Returns
        -------
        Numpy array
            A vector of mean absolute deviations of size m.

        """

        median = np.median(samples, axis=1, keepdims=True)

        abs_diff = np.abs(samples - median)

        mad = np.median(abs_diff, axis=1)

        if scaled:
            return mad * 1.4826  # See Wiki
        else:
            return mad

    @measure_duration()
    def generate_pmnt_distributions(self, card_txns: pd.DataFrame, card_txns_daily: pd.DataFrame):
        """
        Generate distributions of representative values for number of daily
        payment transactions and value of payments. Uses the mean for number
        of payment transactions. We only need a single parameter for number of
        transactions because eventually we use a Poisson distribution to
        sample number of daily transactions. We need two parameters for payment
        value because we eventually draw payment values from a Lognormal
        distribution. We use the median and scaled median absolute deviation
        for representative payment values.
        """
        # Average number of cards transactions
        txns_samples = self.sample_payments(card_txns_daily["txns_w"], m=self.txns_samples_m, n=self.txns_samples_n)
        atxns_distributions = np.mean(txns_samples, axis=1)

        # Average value of payments
        dc_value_samples = self.sample_payments(
            card_txns["amnt_w"][card_txns["card_type"] == "Debit"],
            m=self.value_samples_m,
            n=self.value_samples_n,
        )

        cc_value_samples = self.sample_payments(
            card_txns["amnt_w"][card_txns["card_type"] == "Credit"],
            m=self.value_samples_m,
            n=self.value_samples_n,
        )

        avalue_distributions = pl.DataFrame(
            {
                "dc_means": np.mean(dc_value_samples, axis=1),
                "dc_stds": np.std(dc_value_samples, axis=1),
                "dc_medians": np.median(dc_value_samples, axis=1),
                "dc_mad": Cardsim.calculate_mad(dc_value_samples),
                "cc_means": np.mean(cc_value_samples, axis=1),
                "cc_stds": np.std(cc_value_samples, axis=1),
                "cc_medians": np.median(cc_value_samples, axis=1),
                "cc_mad": Cardsim.calculate_mad(cc_value_samples),
            }
        )
        return atxns_distributions, avalue_distributions

    @staticmethod
    def calculate_tvalue_params(mean, sd, mu=True):
        """
        Calculate lognormal parameters to feed into transaction value
        generator.

        Formulas:
            mu = ln(m^2 / sqrt(m^2 + sd^2))
            sigma = sqrt(ln(1 + (sd^2 / m^2))

        Parameters
        ----------
        mean : float
            The average payment value for a payer.
        sd : float
            The standard deviation of payment values for a payer.
        mu : bool
            True calculates the mu parameter. False calculates the sigma
            parameter. The default is True.

        Returns
        -------
        Numpy array
            Returns a 1-D numpy array

        """

        if mu:
            return np.log(mean**2 / np.sqrt(mean**2 + sd**2))
        else:
            return np.sqrt(np.log(1 + (sd**2 / mean**2)))

    @measure_duration()
    def generate_payer_profiles(self, n_payers: int, atxns_distributions: np.ndarray, avalue_distributions: pd.DataFrame):
        logging.info(f"Generating payer profiles for {n_payers} payers")
        df = pl.DataFrame(
            {
                "payer_id": range(n_payers),
                "payer_x": np.random.randint(0, self.grid_size, n_payers),
                "payer_y": np.random.randint(0, self.grid_size, n_payers),
                "mean_frequency": np.random.choice(atxns_distributions, size=n_payers),
            }
        )
        sampled_indices = np.random.choice(avalue_distributions.index, size=df.height, replace=True)
        df = df.with_columns(
            [
                pl.Series("debit_mean", avalue_distributions.loc[sampled_indices, "dc_medians"].values),
                pl.Series("debit_sd", avalue_distributions.loc[sampled_indices, "dc_mad"].values),
                pl.Series("credit_mean", avalue_distributions.loc[sampled_indices, "cc_medians"].values),
                pl.Series("credit_sd", avalue_distributions.loc[sampled_indices, "cc_mad"].values),
            ]
        )
        df = df.with_columns(
            [
                (pl.col("debit_mean") * self.debit_fraud_mult).alias("debit_mean_fraud"),
                (pl.col("credit_mean") * self.credit_fraud_mult).alias("credit_mean_fraud"),
            ]
        )

        # Prep vars for lognormal distribution
        df = df.with_columns(
            [
                pl.Series("debit_ln_mu", Cardsim.calculate_tvalue_params(df["debit_mean"], df["debit_sd"], mu=True)),
                pl.Series("credit_ln_mu", Cardsim.calculate_tvalue_params(df["credit_mean"], df["credit_sd"], mu=True)),
                pl.Series("debit_ln_sd", Cardsim.calculate_tvalue_params(df["debit_mean"], df["debit_sd"], mu=False)),
                pl.Series("credit_ln_sd", Cardsim.calculate_tvalue_params(df["credit_mean"], df["credit_sd"], mu=False)),
                pl.Series("debit_ln_mu_fraud", Cardsim.calculate_tvalue_params(df["debit_mean_fraud"], df["debit_sd"], mu=True)),
                pl.Series("credit_ln_mu_fraud", Cardsim.calculate_tvalue_params(df["credit_mean_fraud"], df["credit_sd"], mu=True)),
                (pl.col("debit_mean") * pl.col("mean_frequency") * 60).alias("balance"),
            ]
        )
        return df

    @measure_duration()
    def generate_payee_profiles(self, payers: pl.DataFrame):
        """
        Generate payee profiles.
        """
        n_payees = int(payers.height / self.payer_payee_factor)
        logging.info(f"Generating payee profiles for {n_payees} payees")
        payees = pl.DataFrame(
            {
                "payee_id": range(n_payees),
                "payee_x": np.random.randint(0, self.grid_size, n_payees),
                "payee_y": np.random.randint(0, self.grid_size, n_payees),
            }
        )
        return payees, n_payees

    @measure_duration()
    def calculate_distances(self, payers: pl.DataFrame, payees: pl.DataFrame):
        """
        Calculate the distance matrix between payers and payees and related
        components.

        Methodology:
        - Transform payer vector of length n into matrix of shape (n, 1)
        - Subtract payee vector of length m from payer matrix
        - Obtain matrix of shape (n, m)
        - Each element (i, j) is the difference between payer i and payee j
        - Overlay Euclidean distance calc: sqrt((x1 - x2)^2 + (y1 - y2)^2)
        - Ultimately, each entry is distance between payer i and payee j
        - Finally, convert to a long data frame with fields [payer, payee, distance, payee_order]
        """
        # Possible optimization: only calculate the distances for the required pairs instead of the full matrix
        payer_x = payers["payer_x"].to_numpy()[:, None]  # shape (n, 1)
        payer_y = payers["payer_y"].to_numpy()[:, None]  # shape (n, 1)
        payee_x = payees["payee_x"].to_numpy()  # Shape (m, )
        payee_y = payees["payee_y"].to_numpy()  # Shape (m, )
        n_payers = payers.height
        n_payees = payees.height
        distance_matrix = np.sqrt((payer_x - payee_x) ** 2 + (payer_y - payee_y) ** 2)  # Shape (n, m)
        payer_ids = np.repeat(np.arange(n_payers), n_payees)
        payee_ids = np.tile(np.arange(n_payees), n_payers)
        distances = distance_matrix.ravel().astype(np.float32)
        return (
            pl.DataFrame({"payer_id": payer_ids, "payee_id": payee_ids, "distance": distances})
            .sort(by=["payer_id", "distance"], nulls_last=True)
            .with_columns(payee_order=pl.col("payer_id").cum_count().over("payer_id") - 1)
        )

    @measure_duration()
    def generate_baseline_transactions(self, payers: pl.DataFrame, n_days: int, start_date: str):
        """
        Generate the baseline transactions for the simulator. Produces a
        dataframe with payer IDs and dates corresponding to each transaction.

        Parameters
        ----------
        n_days : int
            Number of days the simulator should run.
        start_date : str
            Fictional start date for the simulator in the format YYYY-MM-DD.

        Returns
        -------
        pd.DataFrame
            A data frame of transactions, where the number of rows corresponds
            to the number of transactions.
        """
        start = datetime.fromisoformat(start_date)
        dates_df = pl.DataFrame(
            {
                "day_index": np.arange(n_days),
                "date": pl.date_range(start, end=start + timedelta(days=n_days - 1), interval="1d", eager=True),
            }
        )
        # Cross-join so that each payer is associated with each date
        dates_payers = dates_df.join(payers.select("payer_id", "mean_frequency"), how="cross")
        # The number of transactions in a day, drawn from Poisson, and only retain observations that have transactions
        dates_payers = dates_payers.with_columns(n_txn=np.random.poisson(dates_payers["mean_frequency"])).filter(pl.col("n_txn") > 0)

        # Explode the dataframe based on the number of transactions. Resulting
        # number of (non-unique) payer-date combinations should correspond to n_txn
        return dates_payers.with_columns(pl.col("n_txn").repeat_by("n_txn")).explode("n_txn").select("day_index", "date", "payer_id")

    def calculate_cp_complement(self, p_x: np.ndarray, p_x_given_fraud: np.ndarray) -> np.ndarray:
        """Calculate the complement of the conditional probability for a given
        feature, P(X | !F), using the law of total probability.

        Parameters
        ----------
        p_x : np.ndarray
            Probability of X, P(X)
        p_x_given_fraud : np.ndarray
            Probability of X given fraud, P(X | F)

        Returns
        -------
        np.ndarray
            An array of conditional probability complements
        """

        p_x_given_not_fraud = (p_x - (p_x_given_fraud * self.fraud_rate)) / (1 - self.fraud_rate)
        return p_x_given_not_fraud

    @measure_duration()
    def generate_payment_attribute(self, n_samples: int, atype: Literal["credit_card", "remote"] = "credit_card"):
        """
        Generate a card type or location type payment attribute and likelihood
        ratio. Card and location type follow the same generation logic.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        atype : str
            The type of attribute to derive. Current options are 'credit_card'
            and 'remote'. The default is 'credit_card'.

        Returns
        -------
        np.ndarray
            A vector of 0/1 values corresponding to the dummy variable
            attribute. Also populates relevant likelihood ratio container.
        """
        valid_atypes = ["credit_card", "remote"]
        if atype not in valid_atypes:
            raise ValueError("'atype' must be one of: " + ", ".join(valid_atypes))
        if atype == "credit_card":
            mp = self.credit_card_marginal_p
            cp = self.credit_card_conditional_p
        else:
            mp = self.remote_marginal_p
            cp = self.remote_conditional_p

        # pmnt_attribute = np.random.choice([1, 0], size=n_samples, p=[mp, 1 - mp])
        pmnt_attribute = (np.random.random(size=n_samples) > (1 - mp)).astype(int)
        mask = pmnt_attribute == 1
        p_x = np.where(mask, mp, 1 - mp)
        p_x_given_fraud = np.where(mask, cp, 1 - cp)
        p_x_given_not_fraud = self.calculate_cp_complement(p_x, p_x_given_fraud)
        likelihood_ratio = p_x_given_fraud / p_x_given_not_fraud
        likelihood_ratio = np.minimum(likelihood_ratio, self.lr_cap)
        return pmnt_attribute, likelihood_ratio

    @measure_duration()
    def generate_transaction_value(self, df: pl.DataFrame, payers: pl.DataFrame, with_modification: bool):
        """
        Generate transaction values and likelihood ratios.

        Parameters
        ----------
        df : pd.DataFrame
            A data frame of baseline transactions produced by
            `generate_baseline_transactions()`.

        Returns
        -------
        np.ndarray
            A vector of transaction values.
        """
        if "day_index" not in df.columns:
            raise ValueError("'df' should be baseline transactions but is missing a 'day_index' column")
        # Merge the payment amount details
        df = df.join(
            payers.select(
                "payer_id",
                "debit_ln_mu",
                "debit_ln_sd",
                "debit_ln_mu_fraud",
                "credit_ln_mu",
                "credit_ln_sd",
                "credit_ln_mu_fraud",
            ),
            on="payer_id",
            how="left",
        )

        if with_modification:
            # Merge the payee details
            # payee_vars = ["payee_id", "payee_x", "payee_y"]
            # df = pd.merge(df, payees[payee_vars], how="left", on="payee_id")
            # Merge the payer details
            df = df.join(payers.select("payer_id", "payer_x", "payer_y"), on="payer_id", how="left").with_columns(
                payer_delta_x=pl.col("payer_x") - pl.col("payer_x").mean(),
                payer_delta_y=pl.col("payer_y") - pl.col("payer_y").mean(),
            )
            # df["payee_delta_x"] = df["payee_x"] - df["payee_x"].mean() / df["payee_x"].mean()
            # df["payee_delta_y"] = df["payee_y"] - df["payee_y"].mean() / df["payee_y"].mean()

            payer_scaler = MinMaxScaler(feature_range=(0.8, 1.2))  # type: ignore
            payer_scaled = payer_scaler.fit_transform(df[["payer_delta_x", "payer_delta_y"]].to_numpy())
            df = df.with_columns(
                payer_delta_x=payer_scaled[:, 0],
                payer_delta_y=payer_scaled[:, 1],
            )
            # Create a single column pulling debit and credit params, where relevant
            mask = df["credit_card"] == 1
            mu = np.where(mask, df["credit_ln_mu"], df["debit_ln_mu"]) * (df["payer_delta_x"] * df["payer_delta_y"])
            sigma = np.where(mask, df["credit_ln_sd"], df["debit_ln_sd"])
            fraud_mu = np.where(mask, df["credit_ln_mu_fraud"], df["debit_ln_mu_fraud"]) / (df["payer_delta_x"] * df["payer_delta_y"])
        else:
            # Create a single column pulling debit and credit params, where relevant
            mask = df["credit_card"] == 1
            mu = np.where(mask, df["credit_ln_mu"], df["debit_ln_mu"])
            sigma = np.where(mask, df["credit_ln_sd"], df["debit_ln_sd"])
            fraud_mu = np.where(mask, df["credit_ln_mu_fraud"], df["debit_ln_mu_fraud"])

        transaction_value = np.random.lognormal(mu, sigma).astype(np.float32).round(2)
        # Likelihood ratio calculations. Approximating probability with densities using PDF.
        p_x = lognorm.pdf(transaction_value, s=sigma, scale=np.exp(mu))
        p_x_given_fraud = lognorm.pdf(transaction_value, s=sigma, scale=np.exp(fraud_mu))
        p_x_given_not_fraud = self.calculate_cp_complement(p_x, p_x_given_fraud)
        value_likelihood_ratio = p_x_given_fraud / p_x_given_not_fraud
        value_likelihood_ratio = np.minimum(value_likelihood_ratio, self.lr_cap)
        return transaction_value, value_likelihood_ratio

    @measure_duration()
    def generate_add_payee_distance(self, df: pl.DataFrame, n_payees: int, distances: pl.DataFrame):
        """
        Generate the payee distances and add them to the transactions data
        frame.

        Parameters
        ----------
        df : pd.DataFrame
            A data frame of transactions that has had the location type added.

        Returns
        -------
        pd.DataFrame
            The input data frame with the payee distance added.
        """
        if "remote" not in df.columns:
            raise ValueError("'df' needs a location type column")
        # df = df.copy()
        # Set up min, max, and mode indices for triangular distributions
        min_index = 0
        max_index = n_payees - 1  # zero-based indexing
        inperson_mode = self.distance_mode_quantile["in_person"] * max_index
        remote_mode = self.distance_mode_quantile["remote"] * max_index
        fraud_mode = self.distance_mode_quantile["fraud"] * max_index

        # Select mode for triangular distribution based on location type
        mode_vector = np.where(df["remote"] == 1, remote_mode, inperson_mode)
        # Draw payee indices from triangular distribution
        drawn_index = np.random.triangular(left=min_index, mode=mode_vector, right=max_index, size=df.height)

        # Round indices and ensure within bounds
        # Give the variable the same name as the var that will be merged
        df = (
            df.with_columns(payee_order=np.clip(np.round(drawn_index).astype(int), min_index, max_index))
            .join(distances, on=["payer_id", "payee_order"], how="left")
            .drop("payee_order")
        )

        # Distance likelihood ratio
        # Scipy expects: x (value), loc (left), scale (right - left), and c,
        # which is (mode - loc) / scale. Because left is simply 0 in this case,
        # c simplifies to mode / max_index, and scale is simply max_index.
        p_x = triang.pdf(drawn_index, c=mode_vector / max_index, loc=min_index, scale=max_index)
        p_x_given_fraud = triang.pdf(drawn_index, c=fraud_mode / max_index, loc=min_index, scale=max_index)
        p_x_given_not_fraud = self.calculate_cp_complement(p_x, p_x_given_fraud)

        # Working directly with the draws should prevent divide by zero errors.
        # If errors ever emerge, options are: (1) shift scale up by 1, but this
        # would require adjusting the c calculation, (2) working with distances,
        # but that would make the lookup more complicated.

        distance_likelihood_ratio = p_x_given_fraud / p_x_given_not_fraud
        distance_likelihood_ratio = np.minimum(distance_likelihood_ratio, self.lr_cap)
        return df, distance_likelihood_ratio

    @staticmethod
    def calculate_time_density(weights: dict, windows: dict, tri_peak: float) -> np.ndarray:
        """
        Calculate hourly time density for marginal or conditional distributions

        Parameters
        ----------
        weights : dict
            Weights for each component distribution
        windows : dict
            Start and end hours for each window
        tri_peak : float
            Location of the peak for the triangular distribution (as a fraction
            of the day, e.g., 0.5 is noon)

        Returns
        -------
        np.ndarray
            Hourly density values
        """

        hours = np.arange(24)
        density = np.zeros(24)

        # Baseline triangular distribution. Evaluate density at midpoints.
        # (1) Scale hours to [0,1]
        scaled_hours = (hours + 0.5) / 24.0
        tri_dist = triang(c=tri_peak, loc=0, scale=1)
        # (2) Scale triangular density to match window components
        # Dividing by 24 ensures that this sums to 1 over the day
        density += weights["baseline"] * tri_dist.pdf(scaled_hours) / 24.0  # type: ignore

        # Now add peak densities one-by-one
        for window_name, (start, end) in windows.items():
            window_density = np.zeros(24)
            window_width = end - start
            window_mask = (hours >= start) & (hours < end)
            # Uniform density is 1/width in windows and 0 elsewhere
            window_density[window_mask] = 1.0 / window_width
            density += weights[window_name] * window_density

        return density

    @measure_duration()
    def generate_hourly_probabilities(self):
        """Generate marginal and conditional probabilities for each hour."""
        hours = np.arange(24)
        marginal_density = Cardsim.calculate_time_density(
            weights=self.marginal_tod_weights,
            windows=self.marginal_tod_windows,
            tri_peak=0.5,
        )
        conditional_density = Cardsim.calculate_time_density(
            weights=self.conditional_tod_weights,
            windows=self.conditional_tod_windows,
            tri_peak=0.5,
        )
        # Normalize to get a PMF
        marginal_pmf = marginal_density / np.sum(marginal_density)
        conditional_pmf = conditional_density / np.sum(conditional_density)
        if self.tod_smoothing_param is not None:
            conditional_pmf = marginal_pmf * self.tod_smoothing_param + conditional_pmf * (1 - self.tod_smoothing_param)
        df = pl.DataFrame({"hour": hours, "marginal_pmf": marginal_pmf, "conditional_pmf": conditional_pmf})
        return df

    @measure_duration()
    def generate_transaction_time(self, n_samples: int, tod_pmf: pl.DataFrame) -> np.ndarray:
        """
        Generate a vector of times (in seconds) for payment transactions.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        np.ndarray
            A vector of transaction times in seconds.

        """
        hours = np.arange(24)
        probs = tod_pmf["marginal_pmf"].to_numpy()
        # Generate hours using hourly PMF
        selected_hours = np.random.choice(hours, size=n_samples, p=probs)
        # Select random seconds
        selected_seconds = np.random.randint(0, 3600, size=n_samples)
        return selected_hours * 3600 + selected_seconds

    @measure_duration()
    def calculate_tod_likelihood_ratio(self, df: pl.DataFrame, tod_pmf: pl.DataFrame):
        """
        Calculate the time of day likelihood ratios by merging the hourly
        PMFs.

        Parameters
        ----------
        df : pd.DataFrame
            A data frame of transactions that has had time of day features
            added.
        """
        if "hour" not in df.columns:
            raise ValueError("'df' should contain time of day elements")

        df = df.join(tod_pmf, on="hour", how="left")
        p_x = df["marginal_pmf"].to_numpy()
        p_x_given_fraud = df["conditional_pmf"].to_numpy()
        p_x_given_not_fraud = self.calculate_cp_complement(p_x, p_x_given_fraud)
        tod_likelihood_ratio = p_x_given_fraud / p_x_given_not_fraud
        tod_likelihood_ratio = np.minimum(tod_likelihood_ratio, self.lr_cap)
        return tod_likelihood_ratio

    @measure_duration()
    def generate_fraud(
        self,
        distance_likelihood_ratio: np.ndarray,
        card_likelihood_ratio: np.ndarray,
        location_likelihood_ratio: np.ndarray,
        tod_likelihood_ratio: np.ndarray,
        value_likelihood_ratio: np.ndarray,
    ) -> np.ndarray:
        """Generate the fraud flag by ranking posterior odds produced by Bayes'
        rule.

        Returns
        -------
        np.ndarray
            An array of binary values (the fraud flag).
        """
        prior_odds = self.fraud_rate / (1 - self.fraud_rate)

        likelihood_ratio = (
            card_likelihood_ratio * location_likelihood_ratio * value_likelihood_ratio * distance_likelihood_ratio * tod_likelihood_ratio
        )

        posterior_odds = prior_odds * likelihood_ratio
        threshold_odds = np.percentile(posterior_odds, (1 - self.fraud_flag_threshold) * 100)
        fraud_flag = (posterior_odds >= threshold_odds).astype(int)
        return fraud_flag

    @measure_duration()
    def make_transactions_dataframe(self, n_payers: int, n_days: int, start_date: str, with_modification: bool):
        logging.debug("Starting world generation")
        world_start = time.time()
        card_txns, card_txns_daily = self.source_format_dcpc_data()
        txn_dist, value_dist = self.generate_pmnt_distributions(card_txns, card_txns_daily)
        payers = self.generate_payer_profiles(n_payers, txn_dist, value_dist.to_pandas())
        payees, n_payees = self.generate_payee_profiles(payers)
        distances = self.calculate_distances(payers, payees)

        world_runtime = time.time() - world_start
        logging.info(f"Generated world in {world_runtime} seconds")
        logging.info("Starting phase two: generating transactions within world")
        tx_start = time.time()

        df = self.generate_baseline_transactions(payers, n_days=n_days, start_date=start_date)
        credit_card, card_likelihood_ratio = self.generate_payment_attribute(n_samples=df.height, atype="credit_card")
        remote, location_likelihood_ratio = self.generate_payment_attribute(n_samples=df.height, atype="remote")

        df = df.with_columns(credit_card=credit_card, remote=remote)
        amount, value_likelihood_ratio = self.generate_transaction_value(df, payers, with_modification)
        df = df.with_columns(amount=amount)
        df, distance_likelihood_ratio = self.generate_add_payee_distance(df, n_payees, distances)
        tod_pmf = self.generate_hourly_probabilities()

        time_seconds = self.generate_transaction_time(n_samples=df.height, tod_pmf=tod_pmf)
        ms = pl.Series(values=time_seconds * 1000, dtype=pl.Duration("ms"))
        df = df.with_columns(
            timestamp=pl.col("date").cast(pl.Datetime) + ms,
            hour=time_seconds // 3600,
        )

        tod_likelihood_ratio = self.calculate_tod_likelihood_ratio(df, tod_pmf)
        df = df.with_columns(
            fraud=self.generate_fraud(
                distance_likelihood_ratio=distance_likelihood_ratio,
                card_likelihood_ratio=card_likelihood_ratio,
                location_likelihood_ratio=location_likelihood_ratio,
                tod_likelihood_ratio=tod_likelihood_ratio,
                value_likelihood_ratio=value_likelihood_ratio,
            )
        )

        transactions_runtime = time.time() - tx_start
        logging.debug(f"Generated transactions in {transactions_runtime:.2f} seconds")
        return df, payers, payees

    def load(
        self,
        n_payers: int,
        n_days: int,
        start_date: str,
        with_modification: bool,
        cache_dir: str,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        cached_transactions = os.path.join(
            cache_dir, f"transactions-{n_payers}-{n_days}-{start_date}{'-modified' if with_modification else ''}.csv"
        )
        cached_payers = os.path.join(cache_dir, f"payers-{n_payers}.csv")
        cached_payees = os.path.join(cache_dir, f"payees-{int(n_payers / self.payer_payee_factor)}.csv")
        logging.info(f"Loading transactions from {cached_transactions}...")
        try:
            trx = pl.read_csv(cached_transactions, schema=Transaction.schema(with_predicted_label=False))
            payers = pl.read_csv(cached_payers, schema=Payer.schema())
            terminals = pl.read_csv(cached_payees, schema=Terminal.schema())
        except FileNotFoundError:
            logging.info("Cache not found, running simulation...")
            trx, payers, terminals = self.simulate(n_payers, n_days, start_date, with_modification)
            logging.info(f"Simulation complete, caching results to {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)
            trx.write_csv(cached_transactions)
            payers.write_csv(cached_payers)
            terminals.write_csv(cached_payees)
        return trx, payers, terminals

    def simulate(
        self,
        n_payers: int,
        n_days: int,
        start_date: str,
        with_modification: bool,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        trx, payers, terminals = self.make_transactions_dataframe(n_payers, n_days, start_date, with_modification)
        # Create transactions, cards and terminals dataframes with column names matching class attributes.
        trx_schema = Transaction.schema(with_predicted_label=False)
        trx = (
            trx.rename(
                {
                    "payee_id": "terminal_id",
                    "remote": "is_online",
                    "fraud": "is_fraud",
                    "credit_card": "is_credit",
                }
            )
            .select(list(trx_schema.keys()))
            .cast(trx_schema)
        )
        payers_schema = Payer.schema()
        payers = (
            payers.rename(
                {
                    "payer_id": "id",
                    "payer_x": "x",
                    "payer_y": "y",
                }
            )
            .select(list(payers_schema.keys()))
            .cast(payers_schema)
        )
        terminals_schema = Terminal.schema()
        terminals = (
            terminals.rename(
                {
                    "payee_id": "id",
                    "payee_x": "x",
                    "payee_y": "y",
                }
            )
            .select(list(terminals_schema.keys()))
            .cast(terminals_schema)
        )
        return trx, payers, terminals

    # Convenience -------------------------------------------------------------

    def export_transaction_data(self, df: pd.DataFrame, folder: str, csv: bool = True, file_name: Optional[str] = None):
        """Export transaction data to a .csv or a .pkl.

        Parameters
        ----------
        df : pd.DataFrame
            The transaction data
        folder : str
            The name of the destination folder.
        csv : bool, optional
            True exports as a csv. False exports as a serialized pkl.
            The default is True.
        file_name : str or None, optional
            The name of the file. The default is None.
        """

        if not os.path.exists(folder):
            os.makedirs(folder)

        if file_name is None:
            file_name = f"transaction-data-{self.run_id}"
        else:
            file_name = file_name

        path = folder + "/" + file_name
        logging.info(f"Saving data to {path}")
        if csv:
            path = path + ".csv"
            df.to_csv(path, index=False)
        else:
            path = path + ".pkl"
            df.to_pickle(path)
