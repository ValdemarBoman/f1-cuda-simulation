#ifndef EPISODE_MANAGER_H
#define EPISODE_MANAGER_H

#include <vector>
#include "car.h"
#include "track.h"
#include "sac_cuda.h"

struct Episode {
    carState car;
    Track track;
    double lapProgress;
    bool finished;
};

class EpisodeManager {
public:
    EpisodeManager(int numEpisodes);
    ~EpisodeManager();

    void initializeEpisodes();
    void updateEpisodes();
    void resetEpisode(int index);
    void stopEpisodes();

    const std::vector<Episode>& getEpisodes() const;

private:
    int numEpisodes;
    std::vector<Episode> episodes;
};

#endif // EPISODE_MANAGER_H