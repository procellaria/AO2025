import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict
from scipy.stats import norm
import io

def wilson_interval(count, n, confidence=0.95):
    """
    Calcola l'intervallo di confidenza di Wilson per una proporzione.

    Args:
        count: numero di successi
        n: numero totale di prove
        confidence: livello di confidenza (default 0.95)

    Returns:
        (lower, upper): limiti dell'intervallo di confidenza
    """
    if n == 0:
        return 0, 0

    p = count / n
    z = norm.ppf((1 + confidence) / 2)
    z2 = z * z

    # Calcolo dell'intervallo di Wilson
    denominator = 1 + z2/n
    center = (p + z2/(2*n))/denominator
    spread = z * np.sqrt(p*(1-p)/n + z2/(4*n*n))/denominator

    lower = max(0, center - spread)
    upper = min(1, center + spread)

    return lower*100, upper*100

def load_players(use_cleaned=True):
    """Carica i dati dei giocatori dal file Excel."""
    filename = 'data_2.xlsx' if use_cleaned else 'data.xlsx'
    df = pd.read_excel(filename, header=None)

    if not use_cleaned:
        df = df.iloc[1:, :]
        players = [p.strip("'") for p in df[0].tolist()]
    else:
        players = df[0].tolist()

    base_strengths = df[1].tolist()    # Coefficienti base (punti ATP)
    bonuses = df[2].tolist()           # Bonus
    states = df[3].tolist()            # Stati (1 = in gioco, 0 = eliminato)

    # Calcola i coefficienti totali considerando lo stato
    total_strengths = [(s + b) * st for s, b, st in zip(base_strengths, bonuses, states)]

    return players, total_strengths

def play_match(player1, player2, strength1, strength2):
    """Simula una singola partita tra due giocatori."""
    total_strength = strength1 + strength2
    p1 = strength1 / total_strength

    if np.random.random() < p1:
        return player1, strength1
    return player2, strength2

def simulate_round(players, strengths):
    """Simula un singolo turno del torneo."""
    winners = []
    winners_strengths = []
    matches = []  # Lista delle partite giocate in questo turno

    for i in range(0, len(players), 2):
        winner, winner_strength = play_match(
            players[i], players[i+1],
            strengths[i], strengths[i+1]
        )
        winners.append(winner)
        winners_strengths.append(winner_strength)
        matches.append((players[i], players[i+1]))

    return winners, winners_strengths, matches

def simulate_tournament(verbose=True, track_matches=False):
    """Simula un singolo torneo."""
    players, strengths = load_players(use_cleaned=True)
    round_number = 1
    num_players = len(players)

    # Dizionari per tracciare i progressi
    round_reached = defaultdict(int)
    matches_played = []  # Lista di tutte le partite giocate

    if verbose:
        print(f"Inizio torneo con {num_players} giocatori")

    while num_players > 1:
        if verbose:
            print(f"\nTurno {round_number}")
            print("Partite in corso...")

        players, strengths, round_matches = simulate_round(players, strengths)

        # Aggiorna le statistiche
        for p in players:
            round_reached[p] = round_number + 1

        if track_matches:
            matches_played.extend(round_matches)

        num_players = len(players)

        if verbose:
            print("Vincitori del turno:")
            for i, player in enumerate(players, 1):
                print(f"{i}. {player}")

        round_number += 1

    if verbose:
        print(f"\nVincitore del torneo: {players[0]}")

    return players[0], round_reached, matches_played

def calculate_round_probabilities(all_rounds_reached, n_simulations):
    """Calcola le probabilità di raggiungere ogni turno per ogni giocatore."""
    # Definizione dei turni
    rounds_names = {
        4: "Ottavi di finale",
        5: "Quarti di finale",
        6: "Semifinale",
        7: "Finale",
        8: "Vittoria"
    }

    # Inizializza il dizionario delle statistiche
    players_stats = defaultdict(lambda: defaultdict(int))

    # Conta quante volte ogni giocatore ha raggiunto ogni turno
    for rounds in all_rounds_reached:
        for player, round_num in rounds.items():
            for check_round in range(4, 9):
                if round_num >= check_round:
                    players_stats[player][check_round] += 1

    # Converti i conteggi in percentuali
    results = []
    for player in players_stats:
        stats = {rounds_names[r]: (count/n_simulations*100)
                for r, count in players_stats[player].items()}
        results.append((player, stats))

    # Ordina per probabilità di vittoria
    results.sort(key=lambda x: x[1].get("Vittoria", 0), reverse=True)
    return results

def get_round_matches(matches, round_number):
    """
    Estrae le partite di un specifico turno dal totale delle partite.
    round_number: 7 per finale, 6 per semifinali, ecc.
    """
    # Dato che le partite sono memorizzate in ordine cronologico,
    # possiamo calcolare l'indice delle partite di ogni turno
    matches_per_round = {
        7: 1,    # finale
        6: 2,    # semifinali
        5: 4,    # quarti
        4: 8,    # ottavi
        3: 16,   # sedicesimi
        2: 32,   # trentaduesimi
        1: 64    # primo turno
    }

    # Calcola l'indice di inizio per il turno desiderato
    start_idx = sum(matches_per_round[r] for r in range(1, round_number))
    matches_in_round = matches_per_round[round_number]

    # Se non ci sono abbastanza partite, restituisci una lista vuota
    if len(matches) <= start_idx:
        return []

    # Prendi le partite del turno desiderato
    return matches[start_idx:start_idx + matches_in_round]

def calculate_statistics_with_confidence(counts, n_simulations, confidence=0.95):
    """
    Calcola percentuali e intervalli di confidenza per una serie di conteggi.
    """
    stats = []
    for item, count in counts.items():
        percentage = (count/n_simulations) * 100
        lower, upper = wilson_interval(count, n_simulations, confidence)
        stats.append((item, percentage, lower, upper))

    return sorted(stats, key=lambda x: x[1], reverse=True)

def save_statistics_to_file(win_stats, round_probs, final_probs, semifinal_probs, n_simulations):
    """Salva tutte le statistiche con intervalli di confidenza in un file di testo."""
    with open('tennis_statistics.txt', 'w', encoding='utf-8') as f:
        f.write(f"Statistiche torneo basate su {n_simulations} simulazioni\n")
        f.write("Intervalli di confidenza calcolati al 95%\n")
        f.write("=" * 50 + "\n\n")

        # Statistiche complete per ogni giocatore
        f.write("STATISTICHE PER GIOCATORE\n")
        f.write("-" * 30 + "\n\n")

        # Crea un dizionario con tutte le statistiche per giocatore
        player_stats = {}
        for player_data in round_probs:
            player = player_data[0]
            stats = player_data[1]

            # Cerca le statistiche di vittoria per il giocatore
            win_stat = next((w for w in win_stats if w[0] == player), None)

            # Se il giocatore non ha mai vinto, calcola comunque l'intervallo di Wilson
            if win_stat is None:
                victory = 0.0
                lower, upper = wilson_interval(0, n_simulations, 0.95)  # Calcola l'intervallo di Wilson per 0 successi
                victory_ci = (lower, upper)
            else:
                victory = win_stat[1]
                victory_ci = (win_stat[2], win_stat[3])

            player_stats[player] = {
                "rounds": stats,
                "victory": victory,
                "victory_ci": victory_ci
            }

        # Stampa le statistiche ordinate per probabilità di vittoria
        for player, stats in sorted(player_stats.items(),
                                  key=lambda x: x[1]["victory"],
                                  reverse=True):
            f.write(f"\n{player}:\n")
            f.write(f"  Vittoria torneo: {stats['victory']:.2f}% ")
            f.write(f"(CI: [{stats['victory_ci'][0]:.2f}%, {stats['victory_ci'][1]:.2f}%])\n")

            for round_name, prob in stats["rounds"].items():
                f.write(f"  {round_name}: {prob:.2f}%\n")

        # Finali più probabili
        f.write("\n\nFINALI PIÙ PROBABILI\n")
        f.write("-" * 30 + "\n")
        for (p1, p2), prob, lower, upper in final_probs[:10]:
            f.write(f"{p1} vs {p2}: {prob:.2f}% (CI: [{lower:.2f}%, {upper:.2f}%])\n")

        # Semifinali più probabili
        f.write("\n\nSEMIFINALI PIÙ PROBABILI\n")
        f.write("-" * 30 + "\n")
        for (p1, p2), prob, lower, upper in semifinal_probs[:10]:
            f.write(f"{p1} vs {p2}: {prob:.2f}% (CI: [{lower:.2f}%, {upper:.2f}%])\n")

def run_multiple_simulations(n_simulations=1000):
    """Esegue multiple simulazioni e calcola tutte le statistiche con intervalli di confidenza."""
    np.random.seed(int(time.time()))

    winners = []
    all_rounds_reached = []
    all_matches = []

    for i in range(n_simulations):
        if i % 100 == 0:
            print(f"Simulazione {i}/{n_simulations}")
        winner, rounds_reached, matches = simulate_tournament(verbose=False, track_matches=True)
        winners.append(winner)
        all_rounds_reached.append(rounds_reached)
        all_matches.append(matches)

    # Calcola statistiche vittorie con intervalli di confidenza
    win_counts = Counter(winners)
    win_stats = calculate_statistics_with_confidence(win_counts, n_simulations)

    # Calcola statistiche turni raggiunti
    round_probs = calculate_round_probabilities(all_rounds_reached, n_simulations)

    # Calcola probabilità di finali e semifinali con intervalli di confidenza
    final_matches = Counter(tuple(sorted(m)) for matches in all_matches
                          for m in get_round_matches(matches, 7))
    semifinal_matches = Counter(tuple(sorted(m)) for matches in all_matches
                              for m in get_round_matches(matches, 6))

    final_stats = calculate_statistics_with_confidence(final_matches, n_simulations)

    semifinal_stats = calculate_statistics_with_confidence(semifinal_matches, n_simulations)

    # Salva tutte le statistiche su file
    save_statistics_to_file(win_stats, round_probs, final_stats, semifinal_stats, n_simulations)

    # Stampa un sommario a schermo
    print(f"\nRisultati dopo {n_simulations} simulazioni:")
    print("Le statistiche complete sono state salvate nel file 'tennis_statistics.txt'")

    print("\nTop 10 probabilità di vittoria finale (con intervalli di confidenza al 95%):")
    for player, prob, lower, upper in win_stats[:10]:
        print(f"{player}: {prob:.2f}% (CI: [{lower:.2f}%, {upper:.2f}%])")

    print("\nTop 10 finali più probabili (con intervalli di confidenza al 95%):")
    for (p1, p2), prob, lower, upper in final_stats[:10]:
        print(f"{p1} vs {p2}: {prob:.2f}% (CI: [{lower:.2f}%, {upper:.2f}%])")

    return win_stats, round_probs, final_stats, semifinal_stats

def create_web_app():
    st.title("Simulatore Torneo Tennis Australian Open")

    # Sidebar per i parametri
    st.sidebar.header("Parametri Simulazione")
    n_sims = st.sidebar.slider("Numero di simulazioni",
                              min_value=100,
                              max_value=10000,
                              value=1000,
                              step=100)

    # Upload del file Excel
    st.header("Carica i dati dei giocatori")
    uploaded_file = st.file_uploader("Carica il file Excel con i dati dei giocatori",
                                   type=['xlsx'])

    if uploaded_file is not None:
        # Memorizza il file caricato in una variabile di sessione
        if 'excel_data' not in st.session_state:
            st.session_state.excel_data = uploaded_file.getvalue()

        try:
            # Esegui la simulazione
            if st.button("Avvia Simulazione"):
                with st.spinner('Simulazione in corso...'):
                    # Esegui le simulazioni
                    win_stats, round_stats, final_stats, semifinal_stats = run_multiple_simulations(n_sims)

                    # Mostra i risultati
                    st.header("Risultati della simulazione")

                    # Top 10 probabilità di vittoria
                    st.subheader("Top 10 probabilità di vittoria finale")
                    results_df = pd.DataFrame(
                        win_stats[:10],
                        columns=['Giocatore', 'Probabilità (%)', 'CI Lower', 'CI Upper']
                    )
                    st.dataframe(results_df)

                    # Finali più probabili
                    st.subheader("Top 10 finali più probabili")
                    finals_df = pd.DataFrame(
                        [(f"{p1} vs {p2}", prob, lower, upper)
                         for (p1, p2), prob, lower, upper in final_stats[:10]],
                        columns=['Finale', 'Probabilità (%)', 'CI Lower', 'CI Upper']
                    )
                    st.dataframe(finals_df)

                    # Genera il file di statistiche
                    buffer = io.StringIO()
                    save_statistics_to_file(win_stats, round_stats,
                                         final_stats, semifinal_stats,
                                         n_sims, file=buffer)

                    # Offri il download del file completo
                    st.download_button(
                        label="Scarica statistiche complete",
                        data=buffer.getvalue(),
                        file_name="tennis_statistics.txt",
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"Si è verificato un errore: {str(e)}")
    else:
        st.info("Carica un file Excel per iniziare la simulazione")

if __name__ == "__main__":
    create_web_app()    
