from contextlib import redirect_stdout
import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict
from scipy.stats import norm
import io
import graphviz

# Dati hardcoded dal file Excel
INITIAL_DATA = [
    ('Sinner', 11830, 1500, 1),
    ('Jarry', 1390, 0, 0),
    ('Schoolkate', 334, 0, 0),
    ('Daniel', 674, 0, 0),
    ('Giron', 1150, 0, 1),
    ('Hanfmann', 615, 0, 0),
    ('Etcheverry', 1315, 0, 1),
    ('Cobolli', 1512, 0, 0),
    ('Hurkacz', 2555, 0, 0),
    ('Griekspoor', 1280, 0, 0),
    ('Kecmanović', 1021, 0, 1),
    ('Lajović', 742, 0, 0),
    ('Berrettini', 1380, 250, 0),
    ('Norrie', 1082, 0, 0),
    ('Zhang', 1140, 0, 0),
    ('Rune', 2910, 0, 1),
    ('Tsitsipas', 3195, 250, 0),
    ('Michelsen', 1270, 0, 1),
    ('Mccabe', 216, 0, 0),
    ('Landaluce', 414, 0, 0),
    ('Diallo', 646, 0, 1),
    ('Nardi', 637, 0, 0),
    ('Mannarino', 744, 0, 0),
    ('Khachanov', 2410, 0, 1),
    ('Cerúndolo', 1620, 0, 1),
    ('Bublik', 1330, 0, 0),
    ('Díaz Acosta', 714, 0, 1),
    ('Bergs', 783, 150, 0),
    ('Boyer', 452, 0, 0),
    ('Coria', 617, 0, 1),
    ('Van De Zandschulp', 672, 0, 0),
    ('De Minaur', 3535, 0, 1),
    ('Fritz', 5350, 0, 1),
    ('Brooksby', 0, 50, 0),
    ('Ćorić', 639, 0, 0),
    ('Garín', 382, 0, 0),
    ('Comesaña', 662, 0, 0),
    ('Altmaier', 582, 0, 0),
    ('Monfils', 1130, 0, 1),
    ('Mpetshi Perricard', 1651, 150, 0),
    ('Shelton', 2280, 0, 1),
    ('Nakashima', 1335, 100, 0),
    ('Carreño Busta', 317, 0, 0),
    ('Majchrzak', 491, 0, 1),
    ('Bautista Agut', 1067, 0, 0),
    ('Shapovalov', 1006, 0, 0),
    ('Arnaldi', 1305, 0, 0),
    ('Musetti', 2600, 200, 1),
    ('Rublev', 3520, 200, 0),
    ('Fonseca', 520, 500, 0),
    ('Sonego', 1026, 0, 1),
    ('Wawrinka', 371, 50, 0),
    ('Seyboth Wild', 732, 0, 0),
    ('Marozsán', 960, 0, 1),
    ('Rinderknech', 927, 0, 0),
    ('Tiafoe', 2560, 0, 0),
    ('Popyrin', 1840, 0, 0),
    ('Moutet', 772, 0, 1),
    ('Hijikata', 759, 0, 1),
    ('Krueger', 396, 0, 1),
    ('Ugo Carabelli', 624, 0, 0),
    ('Tien', 493, 0, 1),
    ('Samrej', 109, 0, 0),
    ('Medvedev', 5030, 500, 1),
    ('Djokovic', 3900, 2000, 1),
    ('Basavareddy', 566, 0, 0),
    ('Faria', 481, 0, 0),
    ('Kotov', 612, 0, 0),
    ('Onclin', 242, 0, 0),
    ('Opelka', 341, 100, 0),
    ('Nagal', 635, 0, 0),
    ('Macháč', 1805, 0, 1),
    ('Lehečka', 1660, 0, 1),
    ('Tu', 342, 0, 0),
    ('Gaston', 703, 0, 0),
    ('Jasika', 323, 0, 0),
    ('Goffin', 1029, 0, 0),
    ('Bonzi', 819, 150, 1),
    ('Passaro', 629, 0, 0),
    ('Dimitrov', 3200, 0, 0),
    ('Draper', 2530, 0, 1),
    ('Navone', 1173, 0, 0),
    ('Kokkinakis', 766, 0, 0),
    ('Safiullin', 823, 0, 0),
    ('Džumhur', 679, 0, 0),
    ('Vukic', 778, 0, 1),
    ('Klein', 405, 0, 0),
    ('Korda', 2000, 0, 0),
    ('Thompson', 1695, 0, 0),
    ('Koepfer', 485, 0, 0),
    ('Müller', 965, 0, 0),
    ('Borges', 1445, 0, 1),
    ('Nishioka', 807, 0, 0),
    ('Dougaz', 254, 0, 0),
    ('Shevchenko', 743, 0, 0),
    ('Alcaraz', 7010, 1750, 1),
    ('Ruud', 4210, 250, 0),
    ('Munar', 922, 0, 0),
    ('Basilashvili', 273, 0, 1),
    ('Menšík', 1162, 150, 1),
    ('Shang', 1115, 0, 0),
    ('Davidovich Fokina', 790, 0, 1),
    ('Struff', 1240, 0, 0),
    ('Auger Aliassime', 1755, 0, 0),
    ('Tabilo', 1705, 0, 0),
    ('Carballés Baena', 981, 0, 1),
    ('Duckworth', 637, 0, 0),
    ('Stricker', 173, 0, 0),
    ('Nishikori', 743, 0, 0),
    ('Monteiro', 566, 0, 0),
    ("O'Connell", 770, 0, 0),
    ('Paul', 3145, 0, 1),
    ('Humbert', 2765, 0, 1),
    ('Gigante', 403, 0, 0),
    ('Habib', 264, 50, 0),
    ('Bu', 784, 0, 0),
    ('Walton', 636, 0, 0),
    ('Halys', 756, 0, 0),
    ('Virtanen', 627, 0, 0),
    ('Fils', 2280, 0, 1),
    ('Báez', 1690, 0, 0),
    ('Cazaux', 732, 0, 0),
    ('Fearnley', 632, 0, 1),
    ('Kyrgios', 0, 100, 0),
    ('Martínez', 1220, 0, 0),
    ('Darderi', 1198, 0, 0),
    ('Pouille', 575, 0, 0),
    ('Zverev', 7635, 500, 1),
]

def get_initial_data():
    """Restituisce i dati iniziali in formato utilizzabile"""
    players = [p[0] for p in INITIAL_DATA]
    base_strengths = [p[1] for p in INITIAL_DATA]
    default_bonuses = [p[2] for p in INITIAL_DATA]
    default_states = [p[3] for p in INITIAL_DATA]
    return players, base_strengths, default_bonuses, default_states

def get_tournament_sections():
    """Divide i giocatori in 16 sezioni da 8 giocatori ciascuna"""
    players = [p[0] for p in INITIAL_DATA]
    sections = []
    for i in range(0, len(players), 8):
        sections.append(players[i:i+8])
    return sections

def get_first_round_matches():
    """Restituisce gli accoppiamenti del primo turno"""
    players = [p[0] for p in INITIAL_DATA]
    matches = []
    for i in range(0, len(players), 2):
        matches.append((players[i], players[i+1]))
    return matches

def create_tournament_graph(round_probabilities=None):
    """
    Crea un grafico del tabellone del torneo con:
    - Nodi rettangolari
    - Connessioni ad angolo retto
    - Spessore delle linee proporzionale alle probabilità (se fornite)

    Args:
        round_probabilities: lista di tuple (player, stats) con le probabilità per ogni turno
    """
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR', splines='ortho')

    # Definisci lo stile dei nodi
    dot.attr('node', shape='rectangle', style='filled',
             fillcolor='white', width='2', height='0.5')

    # Numero di giocatori per ogni turno
    players_per_round = [128, 64, 32, 16, 8, 4, 2, 1]
    round_names = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "W"]

    # Dizionario per memorizzare le probabilità dei giocatori per ogni turno
    player_probs = {}
    if round_probabilities:
        for player, stats in round_probabilities:
            # Converti le percentuali in decimali
            player_probs[player] = {
                key: float(value) / 100.0 for key, value in stats.items()
            }

    # Crea i nodi per ogni turno
    for round_num, (num_players, round_name) in enumerate(zip(players_per_round, round_names)):
        with dot.subgraph() as s:
            s.attr(rank='same')
            for i in range(num_players):
                node_id = f"R{round_num}_{i}"
                if round_num == 0:
                    # Primo turno: mostra i nomi dei giocatori
                    player_name = INITIAL_DATA[i][0]
                    s.node(node_id, player_name)
                else:
                    # Altri turni: nodi vuoti con il nome del turno
                    s.node(node_id, round_name)

    # Mappa i nomi dei turni alle probabilità
    round_to_prob = {
        0: "Ottavi di finale",
        1: "Quarti di finale",
        2: "Semifinale",
        3: "Finale",
        4: "Vittoria"
    }

    # Crea gli archi tra i turni con spessore proporzionale alle probabilità
    if round_probabilities:
        max_penwidth = 5.0
        min_penwidth = 0.5

        for round_num in range(len(players_per_round)-1):
            players_current = players_per_round[round_num]
            for i in range(0, players_current, 2):
                winner_idx = i // 2

                if round_num == 0:
                    player1 = INITIAL_DATA[i][0]
                    player2 = INITIAL_DATA[i+1][0]

                    prob1 = player_probs.get(player1, {}).get(round_to_prob[round_num], 0)
                    prob2 = player_probs.get(player2, {}).get(round_to_prob[round_num], 0)

                    penwidth1 = min_penwidth + (max_penwidth - min_penwidth) * prob1
                    penwidth2 = min_penwidth + (max_penwidth - min_penwidth) * prob2

                    dot.edge(f"R{round_num}_{i}", f"R{round_num+1}_{winner_idx}",
                            penwidth=str(penwidth1))
                    dot.edge(f"R{round_num}_{i+1}", f"R{round_num+1}_{winner_idx}",
                            penwidth=str(penwidth2))
                else:
                    # Per gli altri turni usiamo uno spessore standard
                    dot.edge(f"R{round_num}_{i}", f"R{round_num+1}_{winner_idx}",
                            penwidth=str(min_penwidth))
                    dot.edge(f"R{round_num}_{i+1}", f"R{round_num+1}_{winner_idx}",
                            penwidth=str(min_penwidth))

    else:
        # Se non ci sono probabilità, usa linee di spessore uniforme
        for round_num in range(len(players_per_round)-1):
            players_current = players_per_round[round_num]
            for i in range(0, players_current, 2):
                winner_idx = i // 2
                dot.edge(f"R{round_num}_{i}", f"R{round_num+1}_{winner_idx}")
                dot.edge(f"R{round_num}_{i+1}", f"R{round_num+1}_{winner_idx}")

    return dot

def calculate_total_strengths(base_strengths, bonuses):
    """Calcola le forze totali dei giocatori senza applicare lo stato"""
    return [s + b for s, b in zip(base_strengths, bonuses)]

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

def update_winner_strength(winner_strength, loser_strength):
    """
    Aggiorna la forza del vincitore in base alla formula specificata.
    Usa la forza corrente del perdente anche se è stato eliminato.

    Args:
        winner_strength: forza del giocatore vincente
        loser_strength: forza del giocatore perdente (anche se eliminato)

    Returns:
        float: nuova forza del vincitore
    """
    if winner_strength <= 0:
        return winner_strength

    strength_ratio = loser_strength / winner_strength
    update = 0.16 * (winner_strength + loser_strength) / (1 + np.exp(-1 * (strength_ratio - 1)))
    return winner_strength + update

def play_match(player1, player2, strength1, strength2, state1, state2):
    """
    Simula una singola partita tra due giocatori, usando gli stati per determinare il vincitore
    ma le forze originali per l'aggiornamento.

    Args:
        player1, player2: nomi dei giocatori
        strength1, strength2: forze totali (base + bonus) senza stato
        state1, state2: stati dei giocatori (0 o 1)
    """
    # Calcola le forze effettive per determinare il vincitore
    effective_strength1 = strength1 * state1
    effective_strength2 = strength2 * state2

    # Caso 1: entrambi i giocatori sono ancora in gioco
    if effective_strength1 > 0 and effective_strength2 > 0:
        total_strength = effective_strength1 + effective_strength2
        p1 = effective_strength1 / total_strength

        if np.random.random() < p1:
            # Usa le forze originali per l'aggiornamento
            updated_strength = update_winner_strength(strength1, strength2)
            return player1, updated_strength
        updated_strength = update_winner_strength(strength2, strength1)
        return player2, updated_strength

    # Caso 2: solo un giocatore è in gioco (l'altro è stato eliminato)
    if effective_strength1 > 0:
        # Usa le forze originali per l'aggiornamento anche se l'avversario è eliminato
        updated_strength = update_winner_strength(strength1, strength2)
        return player1, updated_strength
    if effective_strength2 > 0:
        updated_strength = update_winner_strength(strength2, strength1)
        return player2, updated_strength

    # Caso 3: entrambi i giocatori sono stati eliminati
    if np.random.random() < 0.5:
        return player1, strength1
    return player2, strength2

def simulate_round(players, strengths, states):
    """Simula un singolo turno del torneo."""
    winners = []
    winners_strengths = []
    matches = []  # Lista delle partite giocate in questo turno

    for i in range(0, len(players), 2):
        winner, winner_strength = play_match(
            players[i], players[i+1],
            strengths[i], strengths[i+1],
            states[i], states[i+1]
        )
        winners.append(winner)
        winners_strengths.append(winner_strength)
        matches.append((players[i], players[i+1]))

    return winners, winners_strengths, matches

def simulate_tournament(players, base_strengths, default_bonuses, default_states, verbose=True, track_matches=False):
    """Simula un singolo torneo."""
    current_players = players.copy()
    # Calcola le forze totali senza applicare lo stato
    current_strengths = calculate_total_strengths(
        base_strengths,
        [st.session_state.bonus_modifications.get(p, default_bonus)
         for p, default_bonus in zip(players, default_bonuses)]
    )
    # Mantieni gli stati in un array separato
    current_states = [st.session_state.player_states[p] for p in players]

    round_number = 1
    num_players = len(current_players)
    round_reached = defaultdict(int)
    matches_played = []

    if verbose:
        st.text(f"Inizio torneo con {num_players} giocatori")

    while num_players > 1:
        # Passa gli stati come parametro separato
        current_players, current_strengths, round_matches = simulate_round(
            current_players,
            current_strengths,
            [st.session_state.player_states[p] for p in current_players]
        )

        for p in current_players:
            round_reached[p] = round_number + 1

        if track_matches:
            matches_played.extend(round_matches)

        num_players = len(current_players)
        round_number += 1

    return current_players[0], round_reached, matches_played

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

    # Aggiungi la tabella Top 10 probabilità di vittoria
    output.append("\nTOP 10 PROBABILITÀ DI VITTORIA")
    output.append("-" * 30)
    output.append("\n{:<4} {:<20} {:<12} {:<20}".format(
        "#", "Giocatore", "Prob. (%)", "Int. Confidenza (%)")
    )
    output.append("-" * 60)

    for i, (player, prob, lower, upper) in enumerate(win_stats[:10], 1):
        output.append("{:<4} {:<20} {:<12.2f} [{:.2f}, {:.2f}]".format(
            i, player, prob, lower, upper)
        )

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

    # Tabs per le diverse sezioni dell'interfaccia
    tabs = st.tabs(["Gestione Giocatori", "Simulazione"])

# Tab 1: Gestione Giocatori
    with tabs[0]:
        if 'bonus_modifications' not in st.session_state:
            st.session_state.bonus_modifications = {}
        if 'player_states' not in st.session_state:
            _, _, _, default_states = get_initial_data()
            st.session_state.player_states = {
                player[0]: player[3] for player in INITIAL_DATA
            }

        # Mostra i giocatori raggruppati per sezione
        sections = get_tournament_sections()
        for section_num, section in enumerate(sections, 1):
            st.subheader(f"Sezione {section_num}")

            # Crea un singolo form per sezione
            with st.form(key=f"section_form_{section_num}"):
                cols = st.columns(4)
                for i, player in enumerate(section):
                    col_idx = i % 4
                    with cols[col_idx]:
                        st.write(f"**{player}**")
                        base_strength = next(p[1] for p in INITIAL_DATA if p[0] == player)
                        default_bonus = next(p[2] for p in INITIAL_DATA if p[0] == player)
                        default_state = next(p[3] for p in INITIAL_DATA if p[0] == player)

                        current_bonus = st.session_state.bonus_modifications.get(player, default_bonus)
                        min_allowed_bonus = -base_strength

                        if default_state == 1:
                            new_bonus = st.number_input(
                                "Bonus",
                                value=int(current_bonus),
                                min_value=int(min_allowed_bonus),
                                step=50,
                                format="%d",
                                key=f"bonus_{section_num}_{i}"
                            )

                            is_active = st.toggle(
                                "In gioco",
                                value=bool(st.session_state.player_states[player]),
                                key=f"state_{section_num}_{i}"
                            )
                        else:
                            new_bonus = st.number_input(
                                "Bonus",
                                value=int(current_bonus),
                                min_value=int(min_allowed_bonus),
                                step=50,
                                format="%d",
                                key=f"bonus_{section_num}_{i}",
                                disabled=True
                            )

                            is_active = st.toggle(
                                "In gioco",
                                value=False,
                                key=f"state_{section_num}_{i}",
                                disabled=True
                            )

                        total_points = base_strength + new_bonus
                        st.write(f"Totale punti: {int(total_points)}")

                # Submit button per il form della sezione
                submitted = st.form_submit_button("Salva modifiche sezione")
                if submitted:
                    for i, player in enumerate(section):
                        new_bonus = st.session_state[f"bonus_{section_num}_{i}"]
                        is_active = st.session_state[f"state_{section_num}_{i}"]
                        st.session_state.bonus_modifications[player] = new_bonus
                        st.session_state.player_states[player] = 1 if is_active else 0

    # # Tab 2: Tabellone
    # with tabs[1]:
    #     st.subheader("Accoppiamenti Primo Turno")
    #     matches = get_first_round_matches()
    #     cols = st.columns(4)
    #     for i, (p1, p2) in enumerate(matches):
    #         col_idx = i % 4
    #         with cols[col_idx]:
    #             st.write(f"{p1} vs {p2}")
    #
    #     st.subheader("Struttura Completa del Tabellone")
    #
    #     # Se sono disponibili i risultati della simulazione, usa le probabilità
    #     if 'round_probs' in st.session_state:
    #         dot = create_tournament_graph(st.session_state.round_probs)
    #     else:
    #         dot = create_tournament_graph()
    #
    #     st.graphviz_chart(dot)

    # Tab 3: Simulazione
    with tabs[1]:
        st.sidebar.header("Parametri Simulazione")
        n_sims = st.sidebar.slider("Numero di simulazioni",
                                min_value=1000,
                                max_value=20000,
                                value=4000,
                                step=200)

        if st.button("Avvia Simulazione"):
            with st.spinner('Simulazione in corso...'):
                progress_bar = st.progress(0)

                players, base_strengths, default_bonuses, default_states = get_initial_data()

                # Esegui le simulazioni multiple
                np.random.seed(int(time.time()))

                winners = []
                all_rounds_reached = []
                all_matches = []

                for i in range(n_sims):
                    if i % 100 == 0:
                        progress_bar.progress(i / n_sims)

                    winner, rounds_reached, matches = simulate_tournament(
                        players, base_strengths, default_bonuses, default_states,
                        verbose=False, track_matches=True
                    )
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

                # Salva i risultati nello stato della sessione
                st.session_state.win_stats = win_stats
                st.session_state.round_probs = round_probs
                st.session_state.final_stats = final_stats
                st.session_state.semifinal_stats = semifinal_stats

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
                st.dataframe(results_df, hide_index=True)

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
                st.dataframe(finals_df, hide_index=True)

                # Genera il file di statistiche
                stats_content = save_statistics_to_string(
                    win_stats,
                    round_probs,
                    final_stats,
                    semifinal_stats,
                    n_sims,
                    st.session_state.bonus_modifications
                )

                # Offri il download del file completo
                st.download_button(
                    label="Scarica statistiche complete",
                    data=stats_content,
                    file_name="tennis_statistics.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    create_web_app()
