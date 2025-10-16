import marimo

__generated_with = "0.16.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Empathetic Expectation Management""")
    return


@app.cell
def _():
    import marimo as mo

    import jax
    import jax.numpy as jnp
    from enum import IntEnum
    from memo import memo

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import warnings

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from ipywidgets import interact, fixed
    import ipywidgets as widgets
    import warnings
    return IntEnum, jax, jnp, memo, mo, pd, plt, sns, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Set up likelihood table

    - 5 offer points
    - 2 quality types
    - 2 confidence levels
    """
    )
    return


@app.cell
def _(IntEnum, jax, jnp):
    # class A(IntEnum): Low=0; Medium=1; High=2
    # class Friend(IntEnum): Indiferent=0; Friend=1


    class Confidence(IntEnum):
        Low = 0
        High = 1


    class Quality(IntEnum):
        Low = 0
        High = 1


    offers = 100 * jnp.linspace(1, 5, num=5)


    # q ∈ {0,1}: quality → different mean price depending on quality
    @jax.jit
    def price_given_quality(q):
        return jnp.array([200, 400])[q]


    # c ∈ {0,1}: confidence → different uncertainty (std)
    @jax.jit
    def uncertainty_given_confidence(c):
        return 100 * jnp.array([1.5, 0.5])[c]


    # o ∈ {100, 200, 300, 400, 500}: offer → different likelihoods depending on offer
    # we specify a normal distribution for the likelihood of the offer given the confidence and quality, with mean and std given by price_given_quality (q) and uncertainty_given_confidence (c)
    @jax.jit
    def offer_given_confidence_quality(o, c, q):
        return jax.scipy.stats.norm.pdf(
            o, loc=price_given_quality(q), scale=uncertainty_given_confidence(c)
        )


    p_offer = jax.vmap(
        jax.vmap(offer_given_confidence_quality, in_axes=(None, 0, None)),
        in_axes=(None, None, 0),
    )(offers, jnp.array([0, 1]), jnp.array([0, 1]))
    p_offer /= p_offer.sum()
    p_offer.round(2)
    return (
        Confidence,
        Quality,
        offer_given_confidence_quality,
        offers,
        price_given_quality,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""### The Model: Auctioneer plans their offer dependening on their belief of their painter-friend's belief"""
    )
    return


@app.cell
def _(
    Confidence,
    E,
    Pr,
    Quality,
    chooses,
    exp,
    given,
    imagine,
    jax,
    jnp,
    knows,
    log,
    memo,
    observes,
    offer_given_confidence_quality,
    offers,
    painter,
    thinks,
):
    # type: ignore

    #
    @jax.jit
    def get_beta_prior(q, case):
        # indexed this way bc 0=Low, 1=High
        return jnp.array(
            [
                [1, 1],  # Unsure: uniform prior
                [4, 1],  # High competence / Good quality
                [1, 4],  # Low competence / Bad quality
            ]
        )[case, q]


    @jax.jit
    def get_min(a, b):
        return jnp.minimum(a, b)


    @jax.jit
    def get_max(a, b):
        return jnp.maximum(a, b)


    @memo
    def auctioneer_makes_offer[o: offers, c: Confidence, q: Quality](
        prior_q,
        prior_c,
        honesty,
        trustworthiness,
        perceived_care,
        disappointment,
        elation,
    ):
        auctioneer: knows(c, q)

        auctioneer: thinks[
            # painter knows the true quality
            painter : knows(q),
            painter : thinks[
                # painter believes the auctioneer believes the painting is of quality q
                auctioneer : given(q in Quality, wpp=get_beta_prior(q, prior_q)),
                # painter believes the auctioneer believes the painting is of confidence c
                auctioneer : given(
                    c in Confidence, wpp=get_beta_prior(c, prior_c)
                ),
                # auctioneer's likelihood over discrete offer values given their confidence and quality beliefs
                auctioneer : given(
                    o in offers, wpp=offer_given_confidence_quality(o, c, q)
                ),
            ],
            # painter: given(observed_q in Quality, wpp=Pr[auctioneer.q == observed_q])
        ]
        # auctioneer chooses an offer that maximizes their utility
        # we take log-probability of each term for numerical stability
        # so we can just add all the utility terms up
        # we also exponentiate the raw log-utility values to get positive numbers
        # so we can normalize them into probabilities
        auctioneer: chooses(
            o in offers,
            wpp=exp(
                +honesty
                * log(
                    offer_given_confidence_quality(o, c, q)
                )  # Honesty / naive response; offer depends on true quality and auctioneer's confidence
                + imagine[
                    # auctioneer imagines painter observes the offer the next day
                    painter : observes[auctioneer.o] is o,
                    # rewards offers that make painter believe the auctioneer's quality assessment matches the true quality
                    +trustworthiness * log(painter[Pr[auctioneer.q == q]])
                    # rewards offers that make the painter think the auctioneer believes the painting is high quality
                    + perceived_care * log(painter[E[auctioneer.q]])
                    # painter's disappointment about the offer:
                    # penalize offers that create a gap between true quality and what the painter thinks the auctioneer believes it is
                    # take abs() bc we want about gap, not direction
                    - disappointment
                    * log(abs(get_min(1e-8, q - painter[E[auctioneer.q]])))
                    # painter's elation about the offer:
                    # rewards offers when true quality exceeds what the painter thinks the auctioneer believes
                    + elation * log(get_max(1e-8, q - painter[E[auctioneer.q]])),
                ]
            ),
        )

        return Pr[auctioneer.o == o]


    honesty = 1
    trustworthiness = 1
    perceived_care = 0
    disappointment = 0
    elation = 0
    prior_quality = 0
    prior_confidence = 0

    auctioneer_makes_offer(
        prior_quality,
        prior_confidence,
        honesty,
        trustworthiness,
        perceived_care,
        disappointment,
        elation,
    )[:, Confidence.Low, Quality.Low].round(2)
    return (auctioneer_makes_offer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Plotting the naive response""")
    return


@app.cell
def _(
    Confidence,
    IntEnum,
    Quality,
    auctioneer_makes_offer,
    offers,
    pd,
    plt,
    price_given_quality,
    sns,
    warnings,
):
    class QualityBelief(IntEnum):
        Unsure = 0
        Low = 1
        High = 2


    class ConfidenceBelief(IntEnum):
        Unsure = 0
        Low = 1
        High = 2


    def show_interactive_plot(
        prior_quality,
        prior_confidence,
        honesty,
        trustworthiness,
        perceived_care,
        disappointment,
        elation,
    ):
        plt.close("all")
        fig, axes = plt.subplots(
            len(Confidence), len(Quality), figsize=(8, 7), sharex=True, sharey=True
        )

        for c in Confidence:
            for q in Quality:
                ax = axes[c, q]
                df = pd.DataFrame(
                    {
                        "Offer": offers.round(),
                        "Probability": auctioneer_makes_offer(
                            prior_quality,
                            prior_confidence,
                            honesty,
                            trustworthiness,
                            perceived_care,
                            disappointment,
                            elation,
                        )[:, c, q],
                    }
                )
                sns.barplot(data=df, x="Offer", y="Probability", ax=ax)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Offer ($)", fontsize=15)
                ax.tick_params(axis="both", which="major", labelsize=17)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ax.set_xticklabels(offers.astype(int), rotation=0)

                ax.set_ylabel(
                    f"Auctioneer confidence\n{c.name.title()}", fontsize=17
                )

                if c == 0:
                    ax.set_title(
                        f"Painting quality\n{q.name.title()}=\\${price_given_quality(q)}",
                        fontsize=17,
                    )

        fig.suptitle("Auctioneer p(offer | confidence, quality)", fontsize=20)
        sns.despine()
        plt.tight_layout()
        plt.show()


    def show_interactive_plot_fixed(
        prior_quality,
        honesty,
        trustworthiness,
        perceived_care,
        disappointment,
        elation,
    ):
        show_interactive_plot(
            prior_quality,
            0,
            honesty,
            trustworthiness,
            perceived_care,
            disappointment,
            elation,
        )
    return ConfidenceBelief, QualityBelief, show_interactive_plot


@app.cell
def _(ConfidenceBelief, QualityBelief, mo):
    # Create marimo widgets
    prior_quality_options = {
        f"{option.name} ({option.value})": option for option in QualityBelief
    }
    prior_quality_widget = mo.ui.dropdown(
        options=prior_quality_options,
        value="Low (1)",  # Use the string key, not the enum value
        label="Prior Quality:",
    )

    prior_confidence_options = {
        f"{option.name} ({option.value})": option for option in ConfidenceBelief
    }
    prior_confidence_widget = mo.ui.dropdown(
        options=prior_confidence_options,
        value="Low (1)",  # Use the string key, not the enum value
        label="Prior Confidence:",
    )

    honesty_widget = mo.ui.slider(
        start=0, stop=2, step=0.1, value=1, label="Honesty:"
    )

    trustworthiness_widget = mo.ui.slider(
        start=0, stop=2, step=0.1, value=1, label="Trustworthiness:"
    )

    perceived_care_widget = mo.ui.slider(
        start=0, stop=2, step=0.1, value=0, label="Perceived Care:"
    )

    disappointment_widget = mo.ui.slider(
        start=0, stop=2, step=0.1, value=0, label="Disappointment:"
    )

    elation_widget = mo.ui.slider(
        start=0, stop=2, step=0.1, value=0, label="Elation:"
    )

    # Layout widgets in a nice grid
    widget_grid = mo.hstack(
        [
            mo.vstack(
                [
                    prior_quality_widget,
                    prior_confidence_widget,
                    honesty_widget,
                    trustworthiness_widget,
                ]
            ),
            mo.vstack(
                [perceived_care_widget, disappointment_widget, elation_widget]
            ),
        ]
    )

    # Display the widgets
    mo.md("## Interactive Auctioneer Model")
    widget_grid
    return (
        disappointment_widget,
        elation_widget,
        honesty_widget,
        perceived_care_widget,
        prior_confidence_widget,
        prior_quality_widget,
        trustworthiness_widget,
    )


@app.cell
def _(
    disappointment_widget,
    elation_widget,
    honesty_widget,
    perceived_care_widget,
    prior_confidence_widget,
    prior_quality_widget,
    show_interactive_plot,
    trustworthiness_widget,
):
    show_interactive_plot(
        prior_quality_widget.value,
        prior_confidence_widget.value,
        honesty_widget.value,
        trustworthiness_widget.value,
        perceived_care_widget.value,
        disappointment_widget.value,
        elation_widget.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Plot sophisticated model""")
    return


@app.cell
def _(
    Confidence,
    Quality,
    auctioneer_makes_offer,
    offers,
    pd,
    plt,
    price_given_quality,
    sns,
    warnings,
):
    def plot_p_offers(
        honesty,
        trustworthiness,
        perceived_care,
        disappointment,
        elation,
        save_fig=False,
        close_fig=False,
    ):
        fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
        for c in Confidence:
            for q in Quality:
                ax = axes[c, q]
                df = pd.DataFrame(
                    {
                        "Offer": offers.round(),
                        "Probability": auctioneer_makes_offer(
                            0,
                            0,
                            honesty,
                            trustworthiness,
                            perceived_care,
                            disappointment,
                            elation,
                        )[:, c, q],
                    }
                )
                sns.barplot(data=df, x="Offer", y="Probability", ax=ax)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Offer ($)", fontsize=15)
                ax.tick_params(axis="both", which="major", labelsize=17)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ax.set_xticklabels(offers.astype(int), rotation=0)

                ax.set_ylabel(
                    f"Auctioneer confidence\n{c.name.title()}", fontsize=17
                )

                if c == 0:
                    ax.set_title(
                        f"Painting quality\n{q.name.title()}=${price_given_quality(q)}",
                        fontsize=17,
                    )

        fig.suptitle(f"Auctioneer p(offer | confidence, quality)", fontsize=20)
        sns.despine()
        plt.tight_layout()

        if save_fig:
            fig.savefig(f"figures/{save_fig}.png", dpi=300)
        if close_fig:
            plt.close(fig)
        else:
            plt.show()

        return fig
    return (plot_p_offers,)


@app.cell
def _(mo):
    # Create sliders for all parameters
    honesty_slider = mo.ui.slider(
        start=0, stop=2, step=0.1, value=1, label="Honesty:"
    )

    trustworthiness_slider = mo.ui.slider(
        start=0, stop=2, step=0.1, value=0, label="Trustworthiness:"
    )

    perceived_care_slider = mo.ui.slider(
        start=0, stop=2, step=0.1, value=0, label="Perceived Care:"
    )

    disappointment_slider = mo.ui.slider(
        start=0, stop=2, step=0.1, value=0, label="Disappointment:"
    )

    elation_slider = mo.ui.slider(
        start=0, stop=2, step=0.1, value=0, label="Elation:"
    )

    # Save figure controls
    save_fig_checkbox = mo.ui.checkbox(value=False, label="Save Figure")

    close_fig_checkbox = mo.ui.checkbox(
        value=False, label="Close Figure After Plotting"
    )

    # Create preset buttons for specific scenarios
    preset_1_button = mo.ui.button(
        label="Preset 1: Base Case", on_click=lambda: None
    )

    preset_2_button = mo.ui.button(
        label="Preset 2: High Care/Emotion", on_click=lambda: None
    )

    # Layout the controls
    controls_layout = mo.vstack(
        [
            mo.md("## Plot P Offers Interactive Controls"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            honesty_slider,
                            trustworthiness_slider,
                            perceived_care_slider,
                        ]
                    ),
                    mo.vstack(
                        [
                            disappointment_slider,
                            elation_slider,
                            mo.hstack([save_fig_checkbox, close_fig_checkbox]),
                        ]
                    ),
                ]
            ),
            mo.md("### Presets:"),
            mo.hstack([preset_1_button, preset_2_button]),
        ]
    )

    controls_layout
    return (
        close_fig_checkbox,
        disappointment_slider,
        elation_slider,
        honesty_slider,
        perceived_care_slider,
        preset_1_button,
        preset_2_button,
        save_fig_checkbox,
        trustworthiness_slider,
    )


@app.cell
def _(
    close_fig_checkbox,
    disappointment_slider,
    elation_slider,
    honesty_slider,
    mo,
    perceived_care_slider,
    preset_1_button,
    preset_2_button,
    save_fig_checkbox,
    trustworthiness_slider,
):
    # Handle preset button clicks and parameter updates
    if preset_1_button.value:
        # Preset 1: Base case (honesty=1, trustworthiness=0, perceived_care=0, disappointment=0, elation=0)
        current_honesty = 1
        current_trustworthiness = 0
        current_perceived_care = 0
        current_disappointment = 0
        current_elation = 0
    elif preset_2_button.value:
        # Preset 2: High care/emotion case
        current_honesty = 1
        current_trustworthiness = 0
        current_perceived_care = 2
        current_disappointment = 2
        current_elation = 2
    else:
        # Use current slider values
        current_honesty = honesty_slider.value
        current_trustworthiness = trustworthiness_slider.value
        current_perceived_care = perceived_care_slider.value
        current_disappointment = disappointment_slider.value
        current_elation = elation_slider.value

    # Display current parameter values
    mo.md(f"""
    ### Current Parameters:
    - **Honesty:** {current_honesty}
    - **Trustworthiness:** {current_trustworthiness}
    - **Perceived Care:** {current_perceived_care}
    - **Disappointment:** {current_disappointment}
    - **Elation:** {current_elation}
    - **Save Figure:** {save_fig_checkbox.value}
    - **Close Figure:** {close_fig_checkbox.value}
    """)
    return (
        current_disappointment,
        current_elation,
        current_honesty,
        current_perceived_care,
        current_trustworthiness,
    )


@app.cell
def _(
    close_fig_checkbox,
    current_disappointment,
    current_elation,
    current_honesty,
    current_perceived_care,
    current_trustworthiness,
    plot_p_offers,
    save_fig_checkbox,
):
    # Generate filename if saving
    if save_fig_checkbox.value:
        fig_name = f"honesty={current_honesty}_trustworthiness={current_trustworthiness}_perceivedcare={current_perceived_care}_disappointment={current_disappointment}_elation={current_elation}"
    else:
        fig_name = False

    # Generate the plot
    plot_p_offers(
        current_honesty,
        current_trustworthiness,
        current_perceived_care,
        current_disappointment,
        current_elation,
        save_fig=fig_name,
        close_fig=close_fig_checkbox.value,
    )
    return


if __name__ == "__main__":
    app.run()
