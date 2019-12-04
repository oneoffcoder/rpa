$(document).ready(function() {
    function modifyMp3Links() {
        const anchors = $('a[class="reference external"]').filter(function(index) { 
            let href = $(this).attr('href');
            return href.endsWith('mp3');
        });
        $(anchors).each(function(index, value) {
            let href = $(this).attr('href');
            $(this).click(function() {
                console.log(href);
                var sound = new Howl({
                    src: [href]
                })
                sound.play();
                return false;
            });
        });
    }

    console.log('ready!');
    
    modifyMp3Links();
    
    console.log('done!');
});