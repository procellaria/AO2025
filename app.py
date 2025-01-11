from contextlib import redirect_stdout
import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict
from scipy.stats import norm
import io


# Dati hardcoded dal file Excel
INITIAL_DATA = [
    ('Sinner', 11830, 1500, 1),
    ('Jarry', 1390, 0, 1),
    ('Schoolkate', 334, 0, 1),
    ('Daniel', 674, 0, 1),
    ('Giron', 1150, 0, 1),
    ('Hanfmann', 615, 0, 1),
    ('Etcheverry', 1315, 0, 1),
    ('Cobolli', 1512, 0, 1),
    ('Hurkacz', 2555, 0, 1),
    ('Griekspoor', 1280, 0, 1),
    ('Kecmanović', 1021, 0, 1),
    ('Lajović', 742, 0, 1),
    ('Berrettini', 1380, 250, 1),
    ('Norrie', 1082, 0, 1),
    ('Zhang', 1140, 0, 1),
    ('Rune', 2910, 0, 1),
    ('Tsitsipas', 3195, 250, 1),
    ('Michelsen', 1270, 0, 1),
    ('Mccabe', 216, 0, 1),
    ('Landaluce', 414, 0, 1),
    ('Diallo', 646, 0, 1),
    ('Nardi', 637, 0, 1),
    ('Mannarino', 744, 0, 1),
    ('Khachanov', 2410, 0, 1),
    ('Cerúndolo', 1620, 0, 1),
    ('Bublik', 1330, 0, 1),
    ('Díaz Acosta', 714, 0, 1),
    ('Bergs', 783, 150, 1),
    ('Boyer', 452, 0, 1),
    ('Coria', 617, 0, 1),
    ('Van De Zandschulp', 672, 0, 1),
    ('De Minaur', 3535, 0, 1),
    ('Fritz', 5350, 0, 1),
    ('Brooksby', 0, 50, 1),
    ('Ćorić', 639, 0, 1),
    ('Garín', 382, 0, 1),
    ('Comesaña', 662, 0, 1),
    ('Altmaier', 582, 0, 1),
    ('Monfils', 1130, 0, 1),
    ('Mpetshi Perricard', 1651, 150, 1),
    ('Shelton', 2280, 0, 1),
    ('Nakashima', 1335, 100, 1),
    ('Carreño Busta', 317, 0, 1),
    ('Majchrzak', 491, 0, 1),
    ('Bautista Agut', 1067, 0, 1),
    ('Shapovalov', 1006, 0, 1),
    ('Arnaldi', 1305, 0, 1),
    ('Musetti', 2600, 200, 1),
    ('Rublev', 3520, 200, 1),
    ('Fonseca', 520, 100, 1),
    ('Sonego', 1026, 0, 1),
    ('Wawrinka', 371, 50, 1),
    ('Seyboth Wild', 732, 0, 1),
    ('Marozsán', 960, 0, 1),
    ('Rinderknech', 927, 0, 1),
    ('Tiafoe', 2560, 0, 1),
    ('Popyrin', 1840, 50, 1),
    ('Moutet', 772, 0, 1),
    ('Hijikata', 759, 0, 1),
    ('Krueger', 396, 0, 1),
    ('Ugo Carabelli', 624, 0, 1),
    ('Tien', 493, 0, 1),
    ('Samrej', 109, 0, 1),
    ('Medvedev', 5030, 500, 1),
    ('Djokovic', 3900, 2500, 1),
    ('Basavareddy', 566, 0, 1),
    ('Faria', 481, 0, 1),
    ('Kotov', 612, 0, 1),
    ('Onclin', 242, 0, 1),
    ('Opelka', 341, 100, 1),
    ('Nagal', 635, 0, 1),
    ('Macháč', 1805, 0, 1),
    ('Lehečka', 1660, 0, 1),
    ('Tu', 342, 0, 1),
    ('Gaston', 703, 0, 1),
    ('Jasika', 323, 0, 1),
    ('Goffin', 1029, 0, 1),
    ('Bonzi', 819, 150, 1),
    ('Fognini', 637, 0, 1),
    ('Dimitrov', 3200, 0, 1),
    ('Draper', 2530, 0, 1),
    ('Navone', 1173, 0, 1),
    ('Kokkinakis', 766, 0, 1),
    ('Safiullin', 823, 0, 1),
    ('Džumhur', 679, 0, 1),
    ('Vukic', 778, 0, 1),
    ('Klein', 405, 0, 1),
    ('Korda', 2000, 0, 1),
    ('Thompson', 1695, 0, 1),
    ('Koepfer', 485, 0, 1),
    ('Müller', 965, 0, 1),
    ('Borges', 1445, 0, 1),
    ('Nishioka', 807, 0, 1),
    ('Dougaz', 254, 0, 1),
    ('Shevchenko', 743, 0, 1),
    ('Alcaraz', 7010, 1000, 1),
    ('Ruud', 4210, 500, 1),
    ('Munar', 922, 0, 1),
    ('Basilashvili', 273, 0, 1),
    ('Menšík', 1162, 150, 1),
    ('Shang', 1115, 0, 1),
    ('Davidovich Fokina', 790, 0, 1),
    ('Struff', 1240, 0, 1),
    ('Auger Aliassime', 1755, 0, 1),
    ('Tabilo', 1705, 0, 1),
    ('Carballés Baena', 981, 0, 1),
    ('Duckworth', 637, 0, 1),
    ('Stricker', 173, 0, 1),
    ('Nishikori', 743, 0, 1),
    ('Monteiro', 566, 0, 1),
    ("O'Connell", 770, 0, 1),
    ('Paul', 3145, 0, 1),
    ('Humbert', 2765, 0, 1),
    ('Gigante', 403, 0, 1),
    ('Habib', 264, 0, 1),
    ('Bu', 784, 0, 1),
    ('Walton', 636, 0, 1),
    ('Halys', 756, 0, 1),
    ('Virtanen', 627, 0, 1),
    ('Fils', 2280, 0, 1),
    ('Báez', 1690, 0, 1),
    ('Cazaux', 732, 0, 1),
    ('Fearnley', 632, 0, 1),
    ('Kyrgios', 0, 100, 1),
    ('Martínez', 1220, 0, 1),
    ('Darderi', 1198, 0, 1),
    ('Pouille', 575, 0, 1),
    ('Zverev', 7635, 500, 1),
]

def get_initial_data():
    """Restituisce i dati iniziali in formato utilizzabile"""
    players = [p[0] for p in INITIAL_DATA]
    base_strengths = [p[1] for p in INITIAL_DATA]
    default_bonuses = [p[2] for p in INITIAL_DATA]
    default_states = [p[3] for p in INITIAL_DATA]
    return players, base_strengths, default_bonuses, default_states

def calculate_total_strengths(base_strengths, bonuses, states):
    """Calcola le forze totali dei giocatori"""
    return [(s + b) * st for s, b, st in zip(base_strengths, bonuses, states)]

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

def load_players(uploaded_file, bonus_modifications=None, use_cleaned=True):
    """
    Carica i dati dei giocatori dal file Excel caricato.

    Args:
        uploaded_file: File Excel caricato
        bonus_modifications: Dizionario con modifiche ai bonus {player_name: new_bonus}
        use_cleaned: Flag per il cleaning dei dati
    Returns:
        players: Lista dei nomi dei giocatori
        total_strengths: Lista delle forze totali
        bonuses: Lista dei bonus attuali
        base_strengths: Lista delle forze base (ATP points)
    """
    df = pd.read_excel(uploaded_file, header=None)

    if not use_cleaned:
        df = df.iloc[1:, :]
        players = [p.strip("'") for p in df[0].tolist()]
    else:
        players = df[0].tolist()

    base_strengths = df[1].tolist()    # Coefficienti base (punti ATP)
    bonuses = df[2].tolist()           # Bonus originali
    states = df[3].tolist()            # Stati (1 = in gioco, 0 = eliminato)

    # Applica le modifiche ai bonus se presenti
    if bonus_modifications:
        for i, player in enumerate(players):
            if player in bonus_modifications:
                bonuses[i] = bonus_modifications[player]

    # Calcola i coefficienti totali considerando lo stato
    total_strengths = [(s + b) * st for s, b, st in zip(base_strengths, bonuses, states)]

    return players, total_strengths, bonuses, base_strengths

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

def save_statistics_to_string(win_stats, round_probs, final_stats, semifinal_stats, n_simulations, bonus_modifications=None):
    """
    Salva tutte le statistiche con intervalli di confidenza in una stringa.
    Ritorna la stringa completa invece di scrivere su file.
    """
    output = []
    output.append(f"SIMULAZIONE AUSTRALIAN OPEN 2025")
    output.append(f"Numero di simulazioni: {n_simulations}")

    if bonus_modifications:
        output.append("\nBONUS MODIFICATI:")
        for player, bonus in bonus_modifications.items():
            output.append(f"{player}: {bonus}")

    output.append("=" * 50)
    output.append("\nSTATISTICHE PER GIOCATORE")
    output.append("-" * 30)

    # Crea un dizionario con tutte le statistiche per giocatore
    player_stats = {}
    for player_data in round_probs:
        player = player_data[0]
        stats = player_data[1]

        # Cerca le statistiche di vittoria per il giocatore
        win_stat = next((w for w in win_stats if w[0] == player), None)

        if win_stat is None:
            victory = 0.0
            lower, upper = wilson_interval(0, n_simulations, 0.95)
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
        output.append(f"\n{player}:")
        output.append(f"  Vittoria torneo: {stats['victory']:.2f}% "
                     f"(CI: [{stats['victory_ci'][0]:.2f}%, {stats['victory_ci'][1]:.2f}%])")

        for round_name, prob in stats["rounds"].items():
            output.append(f"  {round_name}: {prob:.2f}%")

    # Finali più probabili
    output.append("\n\nFINALI PIÙ PROBABILI")
    output.append("-" * 30)
    for (p1, p2), prob, lower, upper in final_stats[:10]:
        output.append(f"{p1} vs {p2}: {prob:.2f}% (CI: [{lower:.2f}%, {upper:.2f}%])")

    # Semifinali più probabili
    output.append("\n\nSEMIFINALI PIÙ PROBABILI")
    output.append("-" * 30)
    for (p1, p2), prob, lower, upper in semifinal_stats[:10]:
        output.append(f"{p1} vs {p2}: {prob:.2f}% (CI: [{lower:.2f}%, {upper:.2f}%])")

    return "\n".join(output)

def create_web_app():
    st.title("Simulatore Australian Open 2025")

    # Inizializzazione dello state
    if 'bonus_modifications' not in st.session_state:
        st.session_state.bonus_modifications = {}
    if 'player_states' not in st.session_state:
        _, _, _, default_states = get_initial_data()
        st.session_state.player_states = {
            player[0]: player[3] for player in INITIAL_DATA
        }

    # Sidebar per i parametri
    st.sidebar.header("Parametri Simulazione")
    n_sims = st.sidebar.slider("Numero di simulazioni",
                              min_value=100,
                              max_value=10000,
                              value=1000,
                              step=100)

    # Carica i dati iniziali
    players, base_strengths, default_bonuses, _ = get_initial_data()

    # Interfaccia per la modifica dei bonus e stati
    st.header("Gestione Giocatori")
    st.write("Modifica bonus e stato dei giocatori")

    # Crea una griglia di 4 colonne
    cols = st.columns(4)
    for i, (player, base_strength, default_bonus) in enumerate(zip(players, base_strengths, default_bonuses)):
        col_idx = i % 4
        with cols[col_idx]:
            st.subheader(player)

            # Bonus input
            current_bonus = st.session_state.bonus_modifications.get(player, default_bonus)
            min_allowed_bonus = -base_strength

            new_bonus = st.number_input(
                "Bonus",
                value=int(current_bonus),
                min_value=int(min_allowed_bonus),
                step=10,
                format="%d",
                key=f"bonus_{i}"
            )

            if new_bonus != current_bonus:
                st.session_state.bonus_modifications[player] = new_bonus

            # Stato del giocatore (in gioco/eliminato)
            is_active = st.toggle(
                "In gioco",
                value=bool(st.session_state.player_states[player]),
                key=f"state_{i}"
            )
            st.session_state.player_states[player] = 1 if is_active else 0

            # Mostra il totale punti
            total_points = base_strength + new_bonus
            st.write(f"Totale punti: {int(total_points)}")

            st.divider()

    # Pulsante per resettare bonus e stati
    if st.button("Reset Tutti i Valori"):
        st.session_state.bonus_modifications = {}
        st.session_state.player_states = {
            player[0]: player[3] for player in INITIAL_DATA
        }
        st.rerun()

    # Esegui la simulazione
    if st.button("Avvia Simulazione"):
        with st.spinner('Simulazione in corso...'):
            def simulate_tournament(verbose=True, track_matches=False):
                current_players = players.copy()
                current_strengths = calculate_total_strengths(
                    base_strengths,
                    [st.session_state.bonus_modifications.get(p, default_bonus)
                     for p, default_bonus in zip(players, default_bonuses)],
                    [st.session_state.player_states[p] for p in players]
                )

                round_number = 1
                num_players = len(current_players)

                round_reached = defaultdict(int)
                matches_played = []

                if verbose:
                    st.text(f"Inizio torneo con {num_players} giocatori")

                while num_players > 1:
                    current_players, current_strengths, round_matches = simulate_round(
                        current_players,
                        current_strengths
                    )

                    for p in current_players:
                        round_reached[p] = round_number + 1

                    if track_matches:
                        matches_played.extend(round_matches)

                    num_players = len(current_players)
                    round_number += 1

                return current_players[0], round_reached, matches_played

                    # Esegui le simulazioni multiple
                    np.random.seed(int(time.time()))

                    winners = []
                    all_rounds_reached = []
                    all_matches = []

                    progress_bar = st.progress(0)

                    for i in range(n_sims):
                        if i % 100 == 0:
                            progress_bar.progress(i/n_sims)

                        winner, rounds_reached, matches = simulate_tournament(verbose=False, track_matches=True)
                        winners.append(winner)
                        all_rounds_reached.append(rounds_reached)
                        all_matches.append(matches)

                    progress_bar.progress(1.0)

                    # Calcola tutte le statistiche
                    win_counts = Counter(winners)
                    win_stats = calculate_statistics_with_confidence(win_counts, n_sims)
                    round_probs = calculate_round_probabilities(all_rounds_reached, n_sims)

                    final_matches = Counter(tuple(sorted(m)) for matches in all_matches
                                          for m in get_round_matches(matches, 7))
                    semifinal_matches = Counter(tuple(sorted(m)) for matches in all_matches
                                              for m in get_round_matches(matches, 6))

                    final_stats = calculate_statistics_with_confidence(final_matches, n_sims)
                    semifinal_stats = calculate_statistics_with_confidence(semifinal_matches, n_sims)

                    # Mostra i risultati
                    st.header("Risultati della simulazione")

                    # Top 10 probabilità di vittoria
                    st.subheader("Top 10 probabilità di vittoria finale")
                    results_data = [
                        [i+1, player, f"{prob:.1f}", f"{lower:.1f}", f"{upper:.1f}"]
                        for i, (player, prob, lower, upper) in enumerate(win_stats[:10])
                    ]
                    results_df = pd.DataFrame(
                        results_data,
                        columns=['#', 'Giocatore', 'Probabilità (%)', 'CI Lower (%)', 'CI Upper (%)']
                    )
                    st.dataframe(results_df)

                    # Finali più probabili
                    st.subheader("Top 10 finali più probabili")
                    finals_data = [
                        [i+1, f"{p1} vs {p2}", f"{prob:.1f}", f"{lower:.1f}", f"{upper:.1f}"]
                        for i, ((p1, p2), prob, lower, upper) in enumerate(final_stats[:10])
                    ]
                    finals_df = pd.DataFrame(
                        finals_data,
                        columns=['#', 'Finale', 'Probabilità (%)', 'CI Lower (%)', 'CI Upper (%)']
                    )
                    st.dataframe(finals_df)

                    # Genera il file di statistiche
                    stats_content = save_statistics_to_string(
                        win_stats,
                        round_probs,
                        final_stats,
                        semifinal_stats,
                        n_sims,
                        bonus_modifications
                    )

                    # Offri il download del file completo
                    st.download_button(
                        label="Scarica statistiche complete",
                        data=stats_content,
                        file_name="tennis_statistics.txt",
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"Si è verificato un errore: {str(e)}")
            st.error("Debug info:")
            st.exception(e)
    else:
        st.info("Carica un file Excel per iniziare la simulazione")

if __name__ == "__main__":
    create_web_app()
